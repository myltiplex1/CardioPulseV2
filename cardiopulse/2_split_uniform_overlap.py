import os
import re
import shutil
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_DIR = "chunked_heart_data_v2"   # output of step 1
OUTPUT_DIR = "split_dataset"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def get_source_id(chunk_filename):
    """Recovers the original recording's identity from a chunk filename
    like 'murmur__201108222246_chunk3.wav' -> 'murmur__201108222246'.
    All chunks sharing a source_id MUST end up in the same split."""
    return re.sub(r'_chunk\d+\.wav$', '', chunk_filename)


def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Could not find '{INPUT_DIR}'. Run step 1 first.")
        return

    class_names = [d for d in os.listdir(INPUT_DIR)
                   if os.path.isdir(os.path.join(INPUT_DIR, d))]

    for split in ["train", "val", "test"]:
        for c in class_names:
            os.makedirs(os.path.join(OUTPUT_DIR, split, c), exist_ok=True)

    print("==================================================")
    print(" STRATIFIED SPLIT BY SOURCE FILE")
    print("==================================================\n")

    summary = {}

    for class_name in class_names:
        class_dir = os.path.join(INPUT_DIR, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]

        # Group chunk filenames by the original recording they came from
        groups = {}
        for f in files:
            src_id = get_source_id(f)
            groups.setdefault(src_id, []).append(f)

        source_ids = list(groups.keys())
        print(f"{class_name}: {len(files)} chunks from {len(source_ids)} "
              f"original recordings")

        if len(source_ids) < 3:
            print(f"  WARNING: only {len(source_ids)} source recordings -- "
                  f"too few to split meaningfully. Putting all in train.")
            train_ids, val_ids, test_ids = source_ids, [], []
        else:
            train_ids, temp_ids = train_test_split(
                source_ids, train_size=TRAIN_RATIO, random_state=RANDOM_SEED
            )
            val_frac_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
            if len(temp_ids) >= 2:
                val_ids, test_ids = train_test_split(
                    temp_ids, train_size=val_frac_of_temp,
                    random_state=RANDOM_SEED
                )
            else:
                val_ids, test_ids = temp_ids, []

        split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        counts = {"train": 0, "val": 0, "test": 0}

        for split, ids in split_map.items():
            for src_id in ids:
                for chunk_file in groups[src_id]:
                    src_path = os.path.join(class_dir, chunk_file)
                    dst_path = os.path.join(OUTPUT_DIR, split, class_name,
                                             chunk_file)
                    shutil.copy2(src_path, dst_path)
                    counts[split] += 1

        summary[class_name] = counts

    print("\n--- SPLIT SUMMARY (chunk counts) ---")
    for c, counts in summary.items():
        print(f"{c}: train={counts['train']}, val={counts['val']}, "
              f"test={counts['test']}")
    print(f"\nOutput location: ./{OUTPUT_DIR}/{{train,val,test}}/<class>/")
    print("==================================================")


if __name__ == "__main__":
    main()
