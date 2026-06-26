#!/usr/bin/env python3
"""llm_pick.py — STEP 4 of the v16 grounder (3.10 venv; has Claude via models.py). Shows Claude a
zoomed hand region with NUMBERED candidate-object outlines (from the SAM3 segment-everything pass)
and asks which ONE the named hand is actively MANIPULATING. Returns {choice, name}; choice = the
candidate index, or -1 for N/A (empty hand / worn item / bare hand / camera-rig). The recording
rig is explicitly NOT a task object. Called as a subprocess by ground_v16.py.

Naming is CONSTRAINED to the clip's canonical inventory (manipulable objects only) so the same
object gets the same name every frame AND non-inventory items (the recording camera, worn items)
fall through to N/A automatically.

Run: ../.venv/bin/python perception/llm_pick.py <img1[,img2]> <LEFT|RIGHT> <n_candidates> <names> [feedback]
  <names> = comma-joined canonical inventory names (may be empty)
Prints one JSON line: {"choice": int, "name": str}
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import vlm_call  # noqa: E402

SYS = (
    "You analyze ONE frame of an egocentric (head-mounted) video of a person doing a tabletop "
    "manipulation task. You are shown TWO crops of the SAME zoomed region around the {hand} hand: the "
    "FIRST image is a CLEAN view (no marks) so you can clearly SEE the objects; the SECOND image is the "
    "SAME crop with COLOURED candidate outlines, a RED DOT at the grasp point, and a YELLOW box marking "
    "where the object is gripped. Below the second image is a KEY listing each candidate as "
    "'<number><letter> = what it was segmented as' in that outline's OWN COLOUR (T = text-segmentation, "
    "P = point-prompt, Z = zoomed text-seg). The outlines are NOT numbered on the image — identify each "
    "one by matching its COLOUR to the key. The SAME object may appear as more than one outline. Use the "
    "CLEAN image to see what is really there, then pick which ONE candidate best and most completely fits "
    "the object the {hand} hand is actively MANIPULATING — holding, using as a tool, moving, pouring, or "
    "working on; answer with that candidate's NUMBER.\n"
    "Name it using one of these canonical object names (copy verbatim):\n"
    "  {names}\n"
    "PART vs WHOLE — name the candidate by the part the OUTLINE actually covers AND that the hand grips. "
    "Some names in the list are a WHOLE object (its main, larger body) and others are only a SMALL PART of "
    "that SAME object (a narrow exposed end, or a thin inner piece that normally sits INSIDE the body). If "
    "the hand is wrapped around the main body of the object, name the WHOLE object — do NOT pick a small-part "
    "name. Pick a small-end-part name ONLY when the outline covers JUST that narrow exposed end. Pick an "
    "inner-piece name ONLY when that thin inner piece is clearly SEPARATE from / outside the body (the body "
    "removed); NEVER call a complete, assembled object by an inner-part name. Use the CLEAN image to see the "
    "whole object and judge where the outline sits along it.\n"
    "Any LATER image (if present) is a REFERENCE SHEET showing a CLEAR cropped view of each named object. "
    "Use it to identify the held object — match what is in the hand to the correct reference, so the SAME "
    "physical object always gets the SAME name even when it is transparent or occluded in this frame.\n"
    "CONTAINER RULE — read carefully: a container may hold loose MATERIAL/SUBSTANCE (loose granular "
    "material, powder, or liquid). ONLY in that case add the suffix '<container> with <material> inside'. "
    "A separate TOOL, DEVICE, or OBJECT (a separate tool, device, lid, or another container) is NEVER "
    "'contents' — do NOT write '<container> with <that object> inside'. If the hand holds a plain "
    "container with nothing loose in it, just name the container. Label ONLY the single object THIS "
    "hand is holding; an object held in the OTHER hand, or hovering / pouring above this one, is a "
    "DIFFERENT object and is NOT inside it. Never label loose material on its own while it sits in a "
    "held container.\n"
    "A handheld device or camera the hand is gripping IS a manipulated object — label it from the "
    "list, do not answer N/A for a held device.\n"
    "Judge ONLY this single moment — is the hand FIRMLY HOLDING an object right now (object lifted "
    "out / enclosed in the grasp)? Answer N/A (choice = -1, name 'N/A') if: the hand is empty / "
    "resting / gesturing; the hand is REACHING toward or merely TOUCHING an object that is still "
    "sitting in a rack / tray / on the table (not yet lifted out and held) — reaching is unstable, "
    "treat it as N/A; the point is on the BARE hand / arm / a WORN item (watch, wrist strap); OR "
    "the held object does NOT match any name in the list. Do NOT invent a name not in the list. "
    "Choose ONLY from the numbered candidates shown."
)
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"},
    "choice": {"type": "integer"},
    "name": {"type": "string"}},
    "required": ["reason", "choice", "name"]}

# Chinese translation of SYS (same meaning; the {names} list + the chosen name stay ENGLISH). Used when PROMPT_LANG=zh.
SYS_ZH = (
    "你正在分析一段第一人称（头戴摄像头）视频中的一帧，画面是一个人在做桌面操作任务。你会看到同一放大区域（围绕"
    "{hand}）的两张裁剪图：第一张是干净视图（无标注），便于你清楚看到物体；第二张是同一裁剪图，叠加了带编号的候选"
    "物体轮廓，并在抓取点处有一个红点。每个编号带一个字母——T = 文本分割掩码，P = 点提示掩码，Z = 放大文本分割掩码；"
    "同一物体可能出现为多个轮廓。请用干净图看清真实存在的物体，然后选出哪一个带编号的轮廓最完整、最贴合{hand}正在"
    "主动“操作”的物体——拿着、当工具用、移动、倾倒或正在加工的物体。\n"
    "请用以下规范物体名称之一来命名（逐字复制，必须用英文）：\n"
    "  {names}\n"
    "部分 vs 整体——按轮廓实际覆盖且手所抓握的部位来命名候选物体。列表中有些名称指整个物体（其主体、较大的本体），"
    "有些只指同一物体的一个小部件（窄的外露端，或通常位于本体内部的薄内件）。如果手握住的是物体的主体，就命名整个"
    "物体——不要选小部件名称。仅当轮廓只覆盖那个窄的外露端时，才选外露端部件名称。仅当那个薄内件明显与本体分离/在"
    "本体之外（本体已被取走）时，才选内件名称；绝不要用内件名称称呼一个完整、已组装的物体。用干净图看清整个物体并"
    "判断轮廓位于其何处。\n"
    "后面若有图片，是参考图，展示每个命名物体清晰的裁剪视图。用它辨认手中物体——把手中之物与正确的参考匹配，使同一"
    "实体物体即使在本帧中透明或被遮挡，也始终得到相同名称。\n"
    "容器规则——请仔细阅读：容器可能盛有松散的材料/物质（松散颗粒、粉末或液体）。仅在这种情况下才加后缀 "
    "'<container> with <material> inside'。一个独立的工具、设备或物体（独立的工具、设备、盖子或另一个容器）绝不算"
    "“内容物”——不要写 '<container> with <that object> inside'。如果手握的容器里没有松散物，就只命名该容器。只标注"
    "这只手所握的那一个物体；另一只手所握、或在其上方悬停/倾倒的物体，是不同的物体，不在其内部。绝不要在松散材料仍"
    "盛于手持容器中时单独标注它。\n"
    "手所握持的手持设备或相机本身就是被操作物体——从列表中给它命名，不要对手持设备回答 N/A。\n"
    "只判断这一个瞬间——此刻手是否正在牢固地握持一个物体（物体被取出/被抓握包住）？在以下情况回答 N/A（choice = -1，"
    "name 为 'N/A'）：手是空的/放松/在比划；手正伸向或仅仅触碰一个仍放在架/盘/桌上的物体（尚未取出并握住）——伸手"
    "去够是不稳定的，按 N/A 处理；红点落在裸手/手臂/穿戴物（手表、腕带）上；或所握物体不匹配列表中任何名称。不要"
    "编造列表外的名称。只能从所显示的带编号候选中选择。"
)


def _b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    imgs = sys.argv[1].split(","); hand = sys.argv[2]; n = int(sys.argv[3])
    names = [s.strip() for s in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if s.strip()]
    feedback = sys.argv[5] if len(sys.argv) > 5 else ""
    frames = [_b64(p) for p in imgs]
    namelist = "\n  ".join(f"- {nm}" for nm in names) if names else "(none provided)"
    zh = os.environ.get("PROMPT_LANG") == "zh"
    hd = {"LEFT": "左手", "RIGHT": "右手"}.get(hand, hand) if zh else hand
    sys_txt = (SYS_ZH if zh else SYS).format(hand=hd, names=namelist)
    user = (f"这是{hd}。候选物体编号为 0..{n - 1}。这只手正在操作哪一个？给出它的编号，以及列表中的一个规范名称"
            f"（必须用英文），或 -1 / 'N/A'。" if zh
            else f"The {hand} hand. Candidate objects are numbered 0..{n - 1}. Which one is this hand "
                 f"manipulating? Give its number and a canonical name FROM THE LIST, or -1 / 'N/A'.")
    if feedback:
        user += (f"\n\n注意：{feedback}" if zh else f"\n\nNOTE: {feedback}")
    r = vlm_call(user, frames, sys_txt, SCHEMA, max_tokens=400)
    choice = int(r.get("choice", -1))
    name = (r.get("name") or "").strip()
    # enforce the canonical vocabulary: keep composites ("<container> with <contents> inside"),
    # snap a bare partial to its inventory name, else N/A
    if names and name and name.upper() not in ("N/A", "NA"):
        low = {nm.lower(): nm for nm in names}
        if name.lower() in low:
            name = low[name.lower()]
        elif any(nm.lower() in name.lower() for nm in names):
            pass                                              # composite contains a canonical container -> keep
        else:
            near = [nm for nm in names if name.lower() in nm.lower()]
            name = near[0] if near else "N/A"
    if choice < 0 or choice >= n or name.upper() in ("N/A", "NA", ""):
        choice, name = -1, "N/A"
    print(json.dumps({"choice": choice, "name": name}))


if __name__ == "__main__":
    main()
