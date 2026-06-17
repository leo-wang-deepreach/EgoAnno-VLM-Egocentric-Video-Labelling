import sys; sys.path.insert(0, ".")
import pipeline as P
P.CHUNK_SEC = 40.0                      # force chunking on a short clip for a fast test
vid = "/home/ubuntu/Internship/egoanno/not_for_testing_videos_2/13592f8b-5470-417b-af18-ecf16651797f_grp022_p01of01.mp4"
ep = P.annotate(vid, "out/chunktest/13592f8b.json", workdir="logs/chunktest_13592f8b", max_attempts=2)
print("=== MERGED (global timeline) ===")
for s in ep["segments"]:
    print(f"  [{s['start_sec']:5.1f}-{s['end_sec']:5.1f}]  L:{str(s.get('left',''))[:26]:<26} R:{str(s.get('right',''))[:26]}")
print("meta:", {k: ep["meta"].get(k) for k in ("chunked","n_chunks","chunk_sec","seams_repaired","chunk_offsets")})
print("duration:", ep["duration_sec"], "| n_segments:", len(ep["segments"]))
