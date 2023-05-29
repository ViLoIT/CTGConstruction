from hashlib import sha256
import pandas as pd
import threading
from file_manager import join_path, mkdir_if_not_exist, is_path_exist, write_file
from joern.joern_parser import run_joern_text
from config import JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR


CSV_PATH = "data/jit_vul_dataset/vul_triggering_commit_data_function_level.csv"
CSV_PATH = "data/jit_vul_dataset/vul_clean_commit_data_function_level.csv"

NUMBER_THREAD = 2
REFACTOR = False


def print_progressbar(percent):
    print("[%-100s] %d%%" % ('='*percent, percent))


#  # ro /* */
def refector_text(text):
    if not isinstance(text,str) :
        return text
    res = list()
    for l in text.splitlines():
        if l.startswith("#") or l.startswith("%"):
            res.append("")
            continue
        elif "/*" in l and "*/" in l:
            res.append("")
            continue
        elif '*/' == l.strip() or '/*' == l.strip():
            res.append("")
            continue
        res.append(l)
    return "\n".join(res)


def main():
    datax = pd.read_csv(CSV_PATH, low_memory=False)
    print(datax.shape)
    code_before = 'before'
    code_after = 'after'
    data = datax[datax[code_before].notna()]
    data = data[data[code_after].notna()]
    print(data.shape)
    threads = list()
    count = 0
    percent = 0
    commits = set(datax["commit_id"])
    size = len(commits)
    print(size)
    for cm_id in list(commits)[::1]:
        count += 1
        # continue
        if int(100 * count/size) > percent:
            percent += 1
            print_progressbar(percent)
        output_file_dir = join_path(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR, cm_id)
        rows = data.loc[data["commit_id"] == cm_id].reset_index()
        rows2 = datax.loc[datax["commit_id"] == cm_id].reset_index()
        c = 0
        pad = rows.shape[0]
        for idx, row in rows.iterrows():
            before_output_file_name = f"before.{idx}"
            after_output_file_name = f"after.{idx}"
            before = row[code_before]
            after = row[code_after]
            if 'line_bl' in row: 
                bl =  row['line_bl']
                if isinstance(bl,str):
                    bl_path = join_path(output_file_dir,f"blame.{idx}.txt")
                    if not is_path_exist(bl_path):
                        write_file(bl_path,bl)
            if is_path_exist(join_path(output_file_dir,f"{after_output_file_name}.cpp.edges.json")):
                continue
            if REFACTOR:
                before = refector_text(before)
                after = refector_text(after)
            t1 = threading.Thread(target=run_joern_text, args=(
                before, output_file_dir, before_output_file_name))
            t2 = threading.Thread(target=run_joern_text, args=(
                after, output_file_dir, after_output_file_name))
            t1.start()
            t2.start()
            threads.append(t1)
            threads.append(t2)
            if len(threads) >= NUMBER_THREAD:
                for t in threads:
                    t.join()
                threads.clear()
        for idx, row in rows2.iterrows():
            before = row[code_before]
            after = row[code_after]
            if  isinstance(before,str) and isinstance(after,str):
                continue
            idx =  pad + c
            c += 1
            before_output_file_name = f"before.{idx}"
            after_output_file_name = f"after.{idx}"
            if 'line_bl' in row: 
                bl =  row['line_bl']
                if isinstance(bl,str):
                    bl_path = join_path(output_file_dir,f"blame.{idx}.txt")
                    if not is_path_exist(bl_path):
                        write_file(bl_path,bl)
            if REFACTOR:
                before = refector_text(before)
                after = refector_text(after)
            t1 = threading.Thread(target=run_joern_text, args=(
                before, output_file_dir, before_output_file_name))
            t2 = threading.Thread(target=run_joern_text, args=(
                after, output_file_dir, after_output_file_name))
            t1.start()
            t2.start()
            threads.append(t1)
            threads.append(t2)
            if len(threads) >= NUMBER_THREAD:
                for t in threads:
                    t.join()
                threads.clear()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
