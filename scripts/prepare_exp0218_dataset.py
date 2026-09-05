#!/usr/bin/env python3
"""Freeze an original, CC0, diagnostic set before any model inference."""
import hashlib
import itertools
import json
from pathlib import Path
import struct
from transformers import AutoTokenizer

ROOT = Path("/mnt/d/llm_exp")
MODEL = ROOT / "models/Qwen3-0.6B"
OUT = ROOT / "results/qwen3-block-htp/exp0218"

EN = [
    "The town library opened a small reading room beside the entrance. On Monday, Nora arranged the history books on the upper shelf and placed the maps in a wide drawer. A sign explained that visitors could borrow books for two weeks, while maps had to remain inside the building. After lunch, a student asked for a map of the river. Nora opened the drawer, found the blue folder, and carried it to a table near the window. The student thanked her and began tracing the route with one finger.",
    "A repair workshop received three boxes of replacement parts before breakfast. Each box had a clear label showing the item name and the date of delivery. Ben checked the labels against the paper list, then put the small screws in a metal tray. The largest box contained wooden handles, which he stored beside the back door. When his colleague arrived, Ben explained where everything had been placed. They agreed to finish the broken chair first because its owner planned to collect it that afternoon.",
    "The morning bus usually stopped outside the old cinema. This week, workers were repairing that section of the street, so the driver used a temporary stop beside the post office. A yellow notice showed the new location and the expected end date of the work. Several passengers arrived early to check the change. One woman helped an elderly man carry his bag across the road. When the bus appeared, the people formed a short line and waited for the driver to open the front door.",
    "The school garden had four narrow beds separated by stone paths. In spring, one class planted carrots while another planted beans. The teacher asked each group to record the weather and measure the plants every Friday. A dry week made the soil hard, so the children brought water from a tap near the fence. They poured it slowly around the roots instead of spraying the leaves. By the following month, the beans had climbed above the short wooden supports, and the class added taller poles.",
    "Before moving to a new apartment, Alice sorted her belongings into several piles. She packed the dishes in old newspapers and wrote the word kitchen on each box. Books went into smaller boxes so that nobody would have to lift too much weight at once. Her brother arrived with a van in the afternoon. Together they loaded the heavy furniture first and placed the fragile objects on top. At the new building, a neighbor held the entrance door open while they carried the first chair upstairs.",
    "A community group prepared a short report about the local park. Volunteers counted visitors at three different times of day and asked which facilities they used most often. Many people mentioned the shaded benches, while parents asked for a cleaner playground. The group summarized the answers in a simple table without recording anyone's name. At the next meeting, the organizer explained how the information had been collected. Members decided to send the report to the town office and keep a copy on the notice board.",
    "The baker tried a new bread recipe on a quiet Tuesday morning. She measured the flour carefully and used slightly less water than usual. After mixing the dough, she covered the bowl with a clean cloth and left it near the warm oven. An hour later, the dough had risen enough to shape. She made two small loaves and marked one with a shallow line. When both were cool, she cut a slice from each and compared their texture before writing a note in her recipe book.",
    "A narrow footbridge crossed the stream behind the village. After heavy rain, a fallen branch blocked part of the path leading to it. Two neighbors brought a saw and moved the smaller pieces away from the water. They left the large trunk beside the path until a cart could collect it. A volunteer checked the bridge boards and found that they were still firm. Before leaving, the neighbors placed a warning sign near the muddy bank so that walkers would approach the bridge carefully.",
    "The museum received a collection of old photographs from a family that had lived nearby for many years. Some pictures showed a market square before the road was widened. Others showed children standing outside a small school. A curator placed each photograph in a protective sleeve and copied the notes written on the back. She planned to display a few images beside a modern map. Visitors would then be able to compare familiar streets with the way they looked several generations ago.",
    "During a long train journey, David sat beside a window and watched the fields pass by. He had packed a sandwich, a bottle of water, and a book about birds. At the first large station, a family entered the carriage and placed their bags on the rack. Their youngest child asked when the train would reach the sea. The father checked the timetable and said they still had two hours to travel. David returned to his book as the train slowly left the platform.",
    "Every Saturday, several neighbors cleaned the shared courtyard. One person swept the steps while another gathered dry leaves near the wall. They placed fallen branches in a separate pile because the collection service handled wood differently from ordinary rubbish. After finishing, they sat outside with tea and discussed small repairs. This week, a loose handle on the gate needed attention. A neighbor offered to bring a screwdriver from home, and the others agreed to hold the gate steady while he worked.",
    "A photographer wanted to make a clear picture of a handmade bowl. She placed it on a plain table beside a large window. The direct sunlight made one side too bright, so she hung a thin white curtain to soften the light. She then moved the bowl a little farther from the wall to reduce the dark shadow behind it. After taking several photographs from different angles, she compared them on a screen and chose the image that showed the shape most clearly.",
]
ZH = [
    "镇上的图书馆在入口旁新开了一间阅览室。星期一早上，小林把历史书放在上层书架，又把地图收进宽大的抽屉。门口的告示写明，普通图书可以借走两周，地图只能在馆内查阅。午饭以后，一名学生想找附近河流的地图。小林打开抽屉，找到蓝色文件夹，放到靠窗的桌子上。学生向她道谢，然后用手指沿着地图上的河道慢慢寻找村庄的位置。他把需要的信息记在自己的笔记本里，离开前将地图整齐地交还给工作人员。",
    "修理铺在开门以前收到三箱零件。每个箱子都有标签，写着物品名称和送达日期。小陈拿纸上的清单逐项核对，把小螺丝倒进金属托盘。最大的箱子装着木头把手，他把它放在后门附近。同事到来后，小陈说明了各种材料的位置。两人决定先修好那把松动的椅子，因为主人下午就要来取。检查完椅腿以后，他们找出合适的螺丝，重新固定连接处，并把椅子放在平地上试了几次，确认它不会摇晃。",
    "早班公交车平时停在旧电影院外面。这一周，工人正在修理那段路面，司机便改用邮局旁边的临时站点。黄色通知写着新站点的位置和预计恢复的日期。几名乘客特意提前出门确认路线。一位年轻人帮助老人把沉重的袋子提过马路。公交车开来时，大家排成短队，等司机打开前门。老人上车坐稳以后，把袋子放到脚边，并向刚才帮忙的年轻人点头致谢。车子随后平稳地驶向下一个路口。",
    "学校花园里有四块细长的菜地，中间用石板小路隔开。春天，一个班种下胡萝卜，另一个班种下豆子。老师请每组学生在星期五记录天气并测量植物的高度。连续几天没有下雨，泥土变得干硬，孩子们便从围栏旁的水龙头接水。他们沿着根部慢慢浇灌，没有把水大量洒在叶子上。到了下个月，豆藤已经超过原来的木架，大家又找来更高的竹竿，并用细绳把藤蔓轻轻固定在上面。",
    "搬进新公寓以前，小王把家里的物品分成几堆。她用旧报纸包好碗碟，在纸箱外写上厨房两个字。书装进较小的箱子，免得一个箱子太重，不方便搬动。下午，哥哥开着货车来帮忙。他们先装大件家具，再把易碎物品放在安全的位置。到了新楼门口，一位邻居替他们扶住入口的门。他们把第一把椅子搬上楼，随后按照箱子上的标记，将不同物品分别送到卧室、书房和厨房。",
    "社区小组准备了一份关于附近公园的简短报告。志愿者在一天中的三个时段统计游客人数，并询问大家最常使用哪些设施。不少人提到树荫下的长椅，也有家长希望儿童活动区更加干净。小组把意见整理成简单的表格，没有记录受访者姓名。在下一次会议上，负责人说明了收集信息的方法。成员决定把报告寄给街道办公室，同时在公告栏保留一份副本，方便没有参加会议的居民阅读并提出补充意见。",
    "面包师在一个安静的星期二早晨试做新配方。她仔细称量面粉，用水比平常稍少一些。面团拌好以后，她用干净的布盖住盆子，把它放在温暖的烤箱附近。一个小时后，面团已经膨胀到适合整形的程度。她做了两个小面包，并在其中一个表面划出浅浅的线。等面包完全放凉，她分别切下一片比较内部的质地，又把观察结果记进配方本。第二天，她准备只调整用水量，其他步骤都保持相同。",
    "村庄后面的小溪上有一座狭窄的人行桥。大雨过后，一根落下的树枝挡住了通向桥头的部分小路。两位邻居带来锯子，把较小的枝条移到远离水边的地方。粗大的树干暂时留在路旁，等手推车来搬运。一名志愿者检查桥面，发现木板仍然牢固。离开以前，大家在泥泞的河岸附近放了一块提醒行人慢走的牌子。傍晚，一家人散步经过这里，看见提示后便牵好孩子的手，谨慎地走过小桥。",
    "博物馆收到附近一户人家捐赠的旧照片。这家人已经在当地住了很多年。有些照片拍的是道路拓宽以前的集市广场，另一些照片上是站在小学校门外的孩子。工作人员给每张照片套上保护袋，并抄录背面的文字说明。她打算挑选几张照片，放在现代地图旁边展出。参观者可以将熟悉的街道与几代人以前的样子作比较。为了方便辨认，展板还会标出拍摄方向和大致年代。",
    "在一次漫长的火车旅行中，小周坐在窗边，看着田野不断向后退去。他带了一份三明治、一瓶水和一本介绍鸟类的书。列车停靠第一个大站时，一家人走进车厢，把行李放到架子上。年纪最小的孩子问什么时候能看见大海。父亲查看时刻表，说还需要两个小时。火车慢慢离开站台，小周重新翻开书，读到一种常在湿地附近活动的小鸟。他想到目的地有一片海湾，或许可以在那里看到它。",
    "每到星期六，几位邻居都会一起打扫共用的院子。有人扫台阶，有人收集墙边的干树叶。大家把落下的树枝单独堆放，因为清运人员处理木料的方式与普通垃圾不同。打扫结束后，他们坐下来喝茶，商量需要修理的小地方。这一周，大门把手有些松动。一位邻居主动回家拿螺丝刀，其他人帮忙扶稳大门。把螺丝拧紧以后，他们又开关几次，确认把手已经牢固，门也能顺利关上。",
    "摄影师想给一个手工制作的碗拍张清楚的照片。她把碗放在大窗户旁的素色桌面上。直射的阳光使一侧过亮，她便挂上一层薄白窗帘，让光线变得柔和。随后，她把碗稍微移离墙壁，减少后面的深色阴影。她从几个角度分别拍摄，在屏幕上比较效果，挑出最能表现碗的形状的一张。最后，她保留原始文件，并在记录中写下窗户方向、拍摄时间和桌面位置，方便以后使用相同的布置。",
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "dataset_v1.json"
    if dest.exists():
        raise SystemExit("Frozen dataset exists; refusing overwrite")
    samples = []
    for lang, docs in [("zh", ZH), ("en", EN)]:
        for doc_id, text in enumerate(docs):
            ids = tok.encode(text, add_special_tokens=False)
            for offset in ([0, 16] if doc_id < 8 else [0]):
                assert len(ids) >= offset + 80, (lang, doc_id, len(ids))
                samples.append(dict(id=len(samples), kind="nll",
                    split="full" if doc_id < 8 else "holdout", language=lang,
                    document_id=f"{lang}-{doc_id}", source_text=text,
                    source_token_offset=offset, prompt_ids=ids[offset:offset+64],
                    target_ids=ids[offset+64:offset+80], steps=16))
    extra = {
        "zh": ["请认真阅读问题。", "请直接给出答案。", "不要添加解释。",
               "不要复述题目。", "请保持回答简短。", "只依据题目提供的信息作答。",
               "请检查答案。", "不要添加无关内容。", "无需说明推理过程。",
               "请使用题目要求的格式。", "答案应当明确。", "请注意细节。"],
        "en": ["Read the question carefully.", "Give only the answer.",
               "Do not explain your reasoning.", "Do not repeat the question.",
               "Keep your response concise.", "Use the information in the question.",
               "Check your answer.", "Do not add unrelated information.",
               "Follow the requested format.", "Be precise.", "Answer directly.",
               "No introduction is needed."]}

    def prompt64(question, lang):
        for n in range(len(extra[lang])+1):
            for chosen in itertools.combinations(extra[lang], n):
                prompt = " ".join((question, *chosen))
                rendered = tok.apply_chat_template([dict(role="user", content=prompt)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok.encode(rendered, add_special_tokens=False)
                if len(ids) == 64:
                    return prompt, rendered, ids
        raise ValueError(f"Cannot construct exactly 64 tokens: {question}")

    tasks = [
        ("zh","reading","红盒子里有3支笔，蓝盒子里有5支笔。蓝盒子里有几支笔？只回答数字。","numeric",5),
        ("zh","reading","会议原定周二举行，后来改到周四。会议最终在哪一天举行？只回答星期名称。","text",["周四","星期四","礼拜四"]),
        ("zh","reading","小林负责采购，小王负责记录，小陈负责检查。谁负责记录？只回答姓名。","text",["小王"]),
        ("zh","reading","包裹的收件地址是海松路8号，寄件人是李青。寄件人叫什么？只回答姓名。","text",["李青"]),
        ("zh","arithmetic","计算7加8，结果是多少？只回答数字。","numeric",15),
        ("zh","arithmetic","一盒有12块饼干，吃掉5块，还剩几块？只回答数字。","numeric",7),
        ("zh","arithmetic","每袋有6个苹果，3袋共有几个苹果？只回答数字。","numeric",18),
        ("zh","arithmetic","把20支笔平均分给4个人，每人几支？只回答数字。","numeric",5),
        ("zh","format","请原样输出这两个汉字：苹果。不要输出标点。","text",["苹果"]),
        ("zh","format",'请输出一个JSON对象，唯一字段ok的值为布尔值true。',"json",{"ok":True}),
        ("zh","format","把英文单词cat转换成大写，只输出转换后的单词。","text",["CAT"]),
        ("zh","format",'请输出一个JSON对象，唯一字段n的值为整数3。',"json",{"n":3}),
        ("en","reading","The red box has 3 pens. The blue box has 5 pens. How many pens are in the blue box? Answer with a number.","numeric",5),
        ("en","reading","The meeting was planned for Tuesday, then moved to Thursday. On which day will it take place? Give the day only.","text",["Thursday"]),
        ("en","reading","Alice buys supplies, Ben takes notes, and Carol checks the work. Who takes notes? Give the name only.","text",["Ben"]),
        ("en","reading","A parcel is addressed to 8 Pine Road. The sender is Nora. What is the sender's name? Give the name only.","text",["Nora"]),
        ("en","arithmetic","What is 7 plus 8? Answer with a number.","numeric",15),
        ("en","arithmetic","A box has 12 biscuits. You eat 5. How many remain? Answer with a number.","numeric",7),
        ("en","arithmetic","Each bag has 6 apples. How many apples are in 3 bags? Answer with a number.","numeric",18),
        ("en","arithmetic","Share 20 pens equally among 4 people. How many pens does each person get? Answer with a number.","numeric",5),
        ("en","format","Output exactly the word apple in lowercase. Do not include punctuation.","text",["apple"]),
        ("en","format","Output a JSON object with the single field ok set to the boolean true.","json",{"ok":True}),
        ("en","format","Convert the word cat to uppercase. Output only the converted word.","text",["CAT"]),
        ("en","format","Output a JSON object with the single field n set to the integer 3.","json",{"n":3}),
    ]
    for lang, category, question, scorer, answer in tasks:
        prompt, rendered, ids = prompt64(question, lang)
        samples.append(dict(id=len(samples), kind="task", split="full", language=lang,
            category=category, prompt=prompt, rendered_prompt=rendered, prompt_ids=ids,
            target_ids=[0]*16, steps=16, scorer=scorer, answer=answer))
    for lang, question in [
        ("zh", "用一句话建议一个适合下雨午后的安静活动。"),
        ("zh", "用一句话说明整理工作笔记的一个好处。"),
        ("en", "Suggest one quiet activity for a rainy afternoon in one sentence."),
        ("en", "Explain one benefit of organizing work notes in one sentence.")]:
        prompt, rendered, ids = prompt64(question, lang)
        samples.append(dict(id=len(samples), kind="open", split="full", language=lang,
            prompt=prompt, rendered_prompt=rendered, prompt_ids=ids, target_ids=[0]*16, steps=16))
    full = [s for s in samples if s["split"] == "full"]
    nll = [s for s in full if s["kind"] == "nll"]
    task = [s for s in full if s["kind"] == "task"]
    quick_nll = [s for lang in ("zh","en") for s in [x for x in nll if x["language"]==lang][::4]]
    quick_task = [task[i] for i in (0,2,4,8,12,14,16,20)]
    dataset = dict(version="qbh-lite-v1", license="CC0-1.0", origin="project-authored",
        limitations="Custom diagnostic set, not a general benchmark. Two overlapping contexts per main document; each document's 32 continuation targets are distinct.",
        construction="Full meaningful chat instructions searched for exactly 64 tokens before inference; no token padding or truncation. Raw-text NLL windows use explicit offsets.",
        tokenizer_sha256=sha(a.model/"qwen3-tokenizer.json"), eos_token_id=tok.eos_token_id,
        quick_ids=[s["id"] for s in quick_nll+quick_task], samples=samples)
    dest.write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n")
    def binary(name, rows):
        vals=[0x51424556,1,len(rows),83]
        for s in rows:
            vals += [s["id"],1 if s["kind"]=="nll" else 2,s["steps"]]+s["prompt_ids"]+s["target_ids"]
        (OUT/name).write_bytes(struct.pack("<"+"I"*len(vals),*vals))
    binary("full.bin",full)
    binary("quick.bin",[dict(s,steps=8) for s in quick_nll]+quick_task)
    binary("holdout.bin",[s for s in samples if s["split"]=="holdout"])
    binary("repeat.bin",[nll[0],nll[16],nll[0],nll[16]])
    binary("smoke.bin",[nll[0],task[0]])
    manifest={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.name.endswith((".json",".bin"))}
    (OUT/"dataset_freeze.json").write_text(json.dumps(manifest,indent=2)+"\n")
    frozen=Path(__file__).resolve().parents[1]/"data/eval/qbh-lite-v1.json"
    frozen.parent.mkdir(parents=True,exist_ok=True)
    frozen.write_bytes(dest.read_bytes())
    print(json.dumps(dict(files=manifest, full_samples=len(full),nll_tokens=32*16, tasks=len(task)),indent=2))

if __name__ == "__main__":
    main()
