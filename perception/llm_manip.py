#!/usr/bin/env python3
"""llm_manip.py — the MANIPULATION GATE (3.10 venv). Decides hold-vs-empty (N/A) with two BROAD yes/no
questions on a clean + overlay crop, voted k times for stability. Replaces the flaky single-pick-or-N/A +
geometry-override. "Manipulating" is intentionally broad (grasp/use/pinch/press/steady), and a SEPARATE
transparent-object question catches clear cups/jars that a plain "holding?" misses.

Run: ../.venv/bin/python perception/llm_manip.py <clean.jpg,overlay.jpg> <LEFT|RIGHT> [k]
Prints one JSON line: {"manip": bool, "q1": int_yes, "q2": int_yes, "k": k}
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import vlm_call  # noqa: E402

SYS = (
    "You see TWO crops of the {hand} hand region in an egocentric (head-mounted) manipulation video: the FIRST "
    "is a CLEAN crop (no marks), the SECOND is the same crop with candidate outlines + a RED grasp dot marking "
    "where THIS hand's fingers grip. Decide whether the {hand} hand is MANIPULATING an object at THIS moment.\n"
    "The object must be IN THIS HAND'S GRASP — at or around the RED dot, held between/under its fingers, in "
    "actual contact. Judge ONLY this {hand} hand and ONLY the object it is gripping.\n"
    "MANIPULATING (broad) = holding, gripping, pinching, using as a tool, picking up, pressing, threading, "
    "assembling, pouring, turning, or STEADYING an object IN ITS GRASP. It does NOT require a firm closed "
    "'hold' — a light grasp or working-on still counts.\n"
    "NOT manipulating = an empty / resting / gesturing hand; a hand HOVERING or REACHING over the table / a "
    "pile / a rack toward an object it has NOT yet grasped; or an object that is merely VISIBLE in the crop "
    "(on the table, in a rack/pile, in the background, or near the hand WITHOUT contact). An object the hand is "
    "not actually touching does NOT count, even if it is large or right beside the hand.\n"
    "Answer TWO yes/no questions:\n"
    "  q1_transparent: is THIS hand GRASPING a TRANSPARENT / clear see-through object (a clear vessel, bottle, "
    "vial, or container) held in its grip at the red dot? Transparent objects are EASY TO MISS, so look "
    "carefully for clear rims, edges, refractions, or contents inside a clear vessel THE HAND IS HOLDING — but "
    "a clear vessel sitting on the table or in a rack that the hand is only reaching toward is NO.\n"
    "  q2_object: is THIS hand grasping ANY object (transparent or not) at the red dot?\n"
    "Be inclusive about light/partial grasps: if the hand is pinching / working on / steadying something small "
    "or partly occluded AT THE GRASP, answer yes. But do NOT answer yes for an object the hand is only near or "
    "reaching toward. Judge ONLY this single moment."
)
SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"},
    "q1_transparent": {"type": "string", "enum": ["yes", "no"]},
    "q2_object": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["reason", "q1_transparent", "q2_object"]}

# Chinese translation of SYS (same meaning; JSON field names + yes/no values stay English). Used when PROMPT_LANG=zh.
SYS_ZH = (
    "你正在观看一段第一人称（头戴摄像头）操作视频中{hand}区域的两张裁剪图：第一张是干净裁剪图（无标注），第二张"
    "是同一裁剪图，叠加了候选物体轮廓和一个红色抓取点，红点标示这只手手指抓握的位置。请判断此刻{hand}是否正在"
    "“操作”某个物体。\n"
    "该物体必须处于这只手的抓握中——位于红点处或其周围，被手指夹住/握住，存在真实接触。只判断这只{hand}，且只判断"
    "它所抓握的物体。\n"
    "“操作”（广义）= 拿着、抓握、捏住、当作工具使用、拿起、按压、穿线、组装、倾倒、转动，或在抓握中稳住某个物体。"
    "它不要求紧握闭合——轻轻抓住或正在加工也算。\n"
    "“非操作” = 空着/放松/在比划的手；在桌面/一堆物体/置物架上方悬停或伸手去够、但尚未抓住物体的手；或仅仅在画面"
    "中可见（在桌上、架上、堆里、背景中，或靠近手但无接触）的物体。手并未真正触碰的物体不算，即使它很大或就在手旁。\n"
    "请回答两个是/否问题：\n"
    "  q1_transparent：这只手是否正在抓握一个透明/清澈可透视的物体（透明的容器、瓶子、小瓶或器皿），并握在红点处的"
    "抓握中？透明物体很容易被忽略，请仔细观察透明的边缘、轮廓、折射，或手所握住的透明容器内的内容物——但若透明容器"
    "只是放在桌上或架上、手只是伸过去够，则为否。\n"
    "  q2_object：这只手是否在红点处抓握任何物体（无论是否透明）？\n"
    "对轻微/部分抓握要从宽：如果手在抓握处捏住/正在加工/稳住某个小的或部分被遮挡的物体，回答 yes。但对于手只是靠近"
    "或伸手去够的物体，不要回答 yes。只判断这一个瞬间。"
)


def _b64(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    imgs = sys.argv[1].split(","); hand = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    frames = [_b64(p) for p in imgs]
    zh = os.environ.get("PROMPT_LANG") == "zh"
    hd = {"LEFT": "左手", "RIGHT": "右手"}.get(hand, hand) if zh else hand
    sys_txt = (SYS_ZH if zh else SYS).format(hand=hd)
    user = (f"这是{hd}。它此刻是否正在操作某个物体？请回答 q1_transparent 和 q2_object。" if zh
            else f"The {hand} hand. Is it manipulating an object right now? Answer q1_transparent and q2_object.")

    def _vote(_):                                            # one independent gate vote (HTTP -> threads OK)
        try:
            r = vlm_call(user, frames, sys_txt, SCHEMA, max_tokens=300)
            return (1 if r.get("q1_transparent", "no").lower() == "yes" else 0,
                    1 if r.get("q2_object", "no").lower() == "yes" else 0)
        except Exception:
            return (0, 0)
    with ThreadPoolExecutor(max_workers=k) as ex:           # k votes CONCURRENTLY (was sequential)
        votes = list(ex.map(_vote, range(k)))
    q1 = sum(v[0] for v in votes); q2 = sum(v[1] for v in votes)
    manip = (q1 * 2 > k) or (q2 * 2 > k)                      # majority on EITHER question
    print(json.dumps({"manip": bool(manip), "q1": q1, "q2": q2, "k": k}))


if __name__ == "__main__":
    main()
