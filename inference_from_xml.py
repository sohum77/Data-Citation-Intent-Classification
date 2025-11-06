# inference_from_xml.py
"""
Usage:
    1) Put one or more XML files into: data/raw/test/XML/
    2) Activate your environment: conda activate bert_env
    3) Run: python inference_from_xml.py

Outputs:
    - models/inference_results/predictions_<filename>.csv   (per-file predictions)
    - models/inference_results/predictions_from_xml.csv     (all combined)
    - Terminal summary (e.g., 65% Primary, 35% Secondary)
"""

import os
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ==== CONFIGURATION ====
MODEL_PATH = "models/bert_baseline_model"             # your trained model
XML_DIR = Path("data/raw/test/XML")                   # folder with XML files
OUT_DIR = Path("models/inference_results")             # where results will be saved
OUT_DIR.mkdir(parents=True, exist_ok=True)             # auto-create if missing

BATCH_SIZE = 8
SENTENCE_MIN_LEN = 20  # ignore very short lines

# ==== SIMPLE SENTENCE SPLITTER ====
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def extract_sentences_from_xml(xml_file):
    """Extracts sentences from XML using BeautifulSoup."""
    sentences = []
    try:
        with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")
    except Exception as e:
        print(f"❌ Could not parse {xml_file}: {e}")
        return []

    text_parts = []
    for tag in ["abstract", "sec", "p", "title", "caption"]:
        for t in soup.find_all(tag):
            txt = t.get_text(separator=" ", strip=True)
            if txt:
                text_parts.append(txt)

    if not text_parts:  # fallback
        text_parts.append(soup.get_text(separator=" ", strip=True))

    for part in text_parts:
        for sent in SENT_SPLIT.split(part):
            sent = sent.strip()
            if len(sent) >= SENTENCE_MIN_LEN:
                sentences.append(sent)

    return list(dict.fromkeys(sentences))  # remove duplicates

def filter_dataset_sentences(sentences):
    """Keep sentences likely mentioning datasets."""
    keywords = ["dataset", "data", "train", "test", "evaluate", "compare", "benchmark"]
    selected = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return selected if selected else sentences

def predict_sentences(sentences, tokenizer, model, device):
    """Predict Primary/Secondary labels for given sentences."""
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i+BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
    return preds

def main():
    if not XML_DIR.exists():
        print(f"❌ Folder not found: {XML_DIR}")
        return

    xml_files = list(XML_DIR.glob("*.xml"))
    if not xml_files:
        print(f"⚠️ No XML files found in {XML_DIR}")
        return

    print(f"✅ Found {len(xml_files)} XML file(s).")
    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cpu")

    all_rows = []
    id2label = {0: "Primary", 1: "Secondary"}

    for xml_file in xml_files:
        print(f"\n📄 Processing: {xml_file.name}")
        sentences = extract_sentences_from_xml(xml_file)
        sentences = filter_dataset_sentences(sentences)

        if not sentences:
            print("  ⚠️ No valid sentences found.")
            continue

        preds = predict_sentences(sentences, tokenizer, model, device)
        df = pd.DataFrame({
            "file": xml_file.name,
            "sentence": sentences,
            "pred_label_id": preds,
            "pred_label": [id2label[p] for p in preds]
        })

        out_file = OUT_DIR / f"predictions_{xml_file.stem}.csv"
        df.to_csv(out_file, index=False)
        print(f"  💾 Saved results → {out_file}")

        # Summary
        counts = df["pred_label"].value_counts().to_dict()
        total = len(df)
        primary = counts.get("Primary", 0)
        secondary = counts.get("Secondary", 0)
        print(f"  📊 Summary for {xml_file.name}:")
        print(f"     Primary:   {primary} ({primary/total*100:.1f}%)")
        print(f"     Secondary: {secondary} ({secondary/total*100:.1f}%)")

        all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_out = OUT_DIR / "predictions_from_xml.csv"
        combined.to_csv(combined_out, index=False)
        print(f"\n✅ Combined predictions saved → {combined_out}")
    else:
        print("⚠️ No predictions generated.")

if __name__ == "__main__":
    main()
