import os
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def main():
    name=os.getenv("CLASSIFIER_MODEL","distilbert/distilbert-base-uncased")
    ds=load_dataset("json",data_files={"train":"data/processed/train.jsonl","validation":"data/processed/validation.jsonl"})
    labels=sorted(set(ds["train"]["category"])); l2i={x:i for i,x in enumerate(labels)}; tok=AutoTokenizer.from_pretrained(name)
    def enc(r):
        x=tok(r["input"],truncation=True,max_length=256,padding="max_length"); x["label"]=l2i[r["category"]]; return x
    ds=ds.map(enc)
    def make(split):
        return tf.data.Dataset.from_tensor_slices(({"input_ids":split["input_ids"],"attention_mask":split["attention_mask"]},split["label"])).batch(8)
    model=TFAutoModelForSequenceClassification.from_pretrained(name,num_labels=len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    model.fit(make(ds["train"]),validation_data=make(ds["validation"]),epochs=3)
    model.save_pretrained("outputs/classifier-tensorflow"); tok.save_pretrained("outputs/classifier-tensorflow")
if __name__=="__main__": main()
