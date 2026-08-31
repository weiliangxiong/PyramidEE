import bisect

def get_equal_split_intervals(step: int, global_max=1024):
    """生成0~global_max等间距分段，返回每个区间下界列表"""
    lows = []
    cur = 0
    while cur <= global_max:
        lows.append(cur)
        cur += step
    return lows

def stat_from_low_to_global_max(sample_lengths: list, step: int, global_max=1024):
    # 排序样本用于二分快速计数
    sorted_samp = sorted(sample_lengths)
    interval_lows = get_equal_split_intervals(step, global_max)
    print(interval_lows)
    res = []
    total_all = len(sorted_samp)
    for low in interval_lows:
        # >= low 的样本数量 = 总样本数 - 小于low的样本数
        count = total_all - bisect.bisect_left(sorted_samp, low)
        res.append(count)
        # res.append({
        #     "区间下界 low": low,
        #     "统计范围": f"[{low}, {global_max}]",
        #     "该范围内样本总数": count
        # })
    return res

# 测试
if __name__ == "__main__":
    import json
    samples = [5, 120, 300, 512, 600, 900, 1024, 200, 88, 15, 666]

    index_path='/data/weilx/compact_cache2/3f1e51ef3c4a_index.json' 

    #index_path='/data/weilx/compact_cache/8e14a477a0c4_index.json'
    with open(index_path, 'r') as f:
            index_data = json.load(f)
        
    seq_lens=[]
    for i, item in enumerate(index_data):
        seq_lens.append(item['seq_len'])


    gap = 50
    result = stat_from_low_to_global_max(seq_lens, gap,global_max=2048)
    
    result= [x / 5489000 for x in result]
    #result= [x / 14472736 for x in result]
    for item in result:
        print(item)