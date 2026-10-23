"""Build SimpleStream's questions_real.json from StreamingBench Real_Time_Visual_Understanding.csv."""
import ast, csv, json, re, sys
from collections import OrderedDict

src, dst = sys.argv[1], sys.argv[2]
videos = OrderedDict()
for row in csv.DictReader(open(src)):
    m = re.search(r"_(sample_\d+)_\d+$", row["question_id"])
    sample = m.group(1)
    entry = videos.setdefault(sample, {"video_path": f"{sample}/video.mp4", "video_categories": "", "questions": []})
    entry["questions"].append({
        "question_id": row["question_id"],
        "task_type": row["task_type"],
        "question": row["question"],
        "time_stamp": row["time_stamp"],
        "answer": row["answer"],
        "options": ast.literal_eval(row["options"]),
        "frames_required": row["frames_required"],
        "temporal_clue_type": row["temporal_clue_type"],
    })
json.dump(list(videos.values()), open(dst, "w"), indent=1, ensure_ascii=False)
print(len(videos), "videos", sum(len(v["questions"]) for v in videos.values()), "questions")
