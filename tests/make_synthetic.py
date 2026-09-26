"""Generate a small synthetic dataset in the challenge format (for smoke tests).

Mimics the noise seen in the real data: legal forms moved/added/dropped,
filler words (Center, Partners), token drops, typos, accents, DBA prefixes,
domains/handles, made-up names, Hindi-script names, reordered/abbreviated
addresses, dropped/zero-padded numbers, placeholders, empty addresses, and
orphan records (entities without an S1 record). Includes a France-like test
country absent from train.

Usage:  python -m tests.make_synthetic --out /tmp/synth --n-entities 4000
"""
import argparse
import random
from pathlib import Path

WORDS = ("apex summit prime global vision care heartland pinnacle cascade harbor "
         "federation rocky electric family committee community green association "
         "super investment classic impex jain management krishna services best "
         "infotech satpura energy nava portillo lewis liquor brogan lloyds cohen "
         "american series pediatric dental specialists house national office "
         "delta tele communication fresh truist nexus anchor rain crystal lending "
         "dream construction kelly advisory orelee barber shop christ chapel").split()
STREETS = "oak maple pine cedar elm vaughn traveler guadalupe darrow mayflower edgewood oakenwald sandy ridge".split()
CITIES = {"US": "phoenix tyler columbus nashville gilbert eastham chicago monroe".split(),
          "India": "patna hyderabad nagpur pune delhi bangalore nashik chennai".split(),
          "France": "lille dunkerque bordeaux roubaix pornic nantes lyon paris".split()}
STATES = {"US": [("AZ", "Arizona"), ("TX", "Texas"), ("OH", "Ohio"), ("TN", "Tennessee")],
          "India": [("MH", "Maharashtra"), ("TG", "Telangana"), ("BR", "Bihar"), ("DL", "Delhi")],
          "France": [("Nord", "Hauts-de-France"), ("Gironde", "Nouvelle-Aquitaine")]}
LEGAL = {"US": ["Inc", "LLC", "Corp", ""], "India": ["Pvt Ltd", "Private Limited", "Limited", ""],
         "France": ["SARL", "SAS", "SCI", ""]}
HINDI = {"super": "सुपर", "investment": "इन्वेस्टमेंट", "care": "केयर", "vision": "विजन",
         "management": "मैनेजमेंट", "jain": "जैन", "krishna": "कृष्ण", "services": "सर्विसेज",
         "best": "बेस्ट", "energy": "एनर्जी", "private": "प्राइवेट", "limited": "लिमिटेड"}
FILLERS = ["Center", "Partners", "Services", "Group"]


def typo(w, r):
    if len(w) < 4:
        return w
    i = r.randrange(1, len(w) - 1)
    return w[:i] + r.choice("aeiourstln") + w[i + 1:]


def make_entity(r, country):
    name = " ".join(w.title() for w in r.sample(WORDS, r.choice([2, 2, 3])))
    legal = r.choice(LEGAL[country])
    num = str(r.randint(1, 9999))
    street = r.choice(STREETS).title() + " " + r.choice(["Road", "Street", "Lane", "Avenue"])
    if country == "France":
        street = "Rue " + r.choice(STREETS).title()
    code, full = r.choice(STATES[country])
    return {"name": (name + " " + legal).strip(), "core": name, "legal": legal, "num": num,
            "street": street, "city": r.choice(CITIES[country]).title(), "code": code, "full": full}


def s1_record(e):
    return e["name"], f"{e['num']} {e['street']}, {e['city']}, {e['code']}"


def noisy_record(e, r, country):
    words = e["core"].split()
    p = r.random()
    if p < 0.04:
        name = r.choice(["Drexavi", "Rizafaye", "Ariadrex", "Haloarc #11217"])
    elif p < 0.09:
        name = "".join(words).lower() + ".com"
    elif p < 0.12:
        name = "Fluxsol DBA " + e["name"]
    elif p < 0.22 and country == "India" and all(w.lower() in HINDI for w in words):
        name = " ".join(HINDI[w.lower()] for w in words) + " " + HINDI["private"] + " " + HINDI["limited"]
    else:
        ws = list(words)
        if r.random() < 0.15 and len(ws) > 2:
            ws.pop(r.randrange(len(ws)))
        if r.random() < 0.15:
            ws[-1] = r.choice(FILLERS)
        if r.random() < 0.15:
            i = r.randrange(len(ws)); ws[i] = typo(ws[i], r)
        if r.random() < 0.1:
            r.shuffle(ws)
        legal = e["legal"] if r.random() < 0.6 else r.choice(LEGAL[country])
        name = (" ".join(ws) + " " + legal) if r.random() < 0.7 else (legal + " " + " ".join(ws))
        if r.random() < 0.3:
            name = name.upper()
        if r.random() < 0.05:
            name = name.replace("a", "á", 1)
    num = e["num"] if r.random() > 0.1 else ""
    if num and r.random() < 0.1:
        num = "0" + num
    street = e["street"].replace("Road", "Rd").replace("Street", "St") if r.random() < 0.5 else e["street"]
    state = e["full"] if r.random() < 0.5 else e["code"]
    parts = [f"{num} {street}".strip(), e["city"], state]
    if r.random() < 0.3:
        r.shuffle(parts)
    if r.random() < 0.05:
        parts.insert(1, "<NULL>")
    addr = ", ".join(parts) if r.random() > 0.03 else ""
    return name.strip(), addr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synth")
    ap.add_argument("--n-entities", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = random.Random(a.seed)
    head = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for split, countries in (("train", ["US", "India"]), ("test", ["US", "India", "France"])):
        d = Path(a.out) / split
        d.mkdir(parents=True, exist_ok=True)
        s1, s2, s3, gt = [], [], [], []
        for i in range(a.n_entities):
            c = r.choice(countries)
            e = make_entity(r, c)
            has_s1 = r.random() > 0.2           # 20% orphan entities (no S1 record)
            k = r.choice([0, 1, 2, 3, 3, 4, 4, 5, 6])
            ids = []
            for j in range(k):
                src = r.choice(["S2", "S3"])
                eid = f"{src}-{r.randrange(10**9)}"
                n, ad = noisy_record(e, r, c)
                (s2 if src == "S2" else s3).append(f"{eid}\t{n}\t{ad}\t{c}\n")
                ids.append(eid)
            # sibling distractor business (no S1): neighbouring number, one word changed
            if r.random() < (0.35 if split == "test" else 0.05) and ids:
                sib = dict(e)
                sib["num"] = str(max(1, int(e["num"]) + r.choice([-1, 1]) * r.randint(1, 30)))
                ws = e["core"].split()
                ws[r.randrange(len(ws))] = r.choice(FILLERS + ["Harbor", "Coastal", "Estate"])
                sib["core"] = " ".join(ws)
                sib["name"] = (sib["core"] + " " + e["legal"]).strip()
                for _ in range(r.choice([1, 2, 3])):
                    src = r.choice(["S2", "S3"])
                    n, ad = noisy_record(sib, r, c)
                    (s2 if src == "S2" else s3).append(f"{src}-{r.randrange(10**9)}\t{n}\t{ad}\t{c}\n")
            if has_s1:
                sid = f"S1-{r.randrange(10**9)}"
                n, ad = s1_record(e)
                s1.append(f"{sid}\t{n}\t{ad}\t{c}\n")
                gt.append(f"{sid}\t{','.join(ids)}\n")
        for name, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
            (d / f"{split}_{name}.tsv").write_text(head + "".join(rows), encoding="utf-8")
        if split == "test":   # hidden truth, only for local evaluation (tests/eval_synthetic.py)
            (Path(a.out) / "test_truth_hidden.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\n" + "".join(gt), encoding="utf-8")
        if split == "train":
            (d / "train_ground_truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\n" + "".join(gt), encoding="utf-8")
        print(split, len(s1), len(s2), len(s3))


if __name__ == "__main__":
    main()
