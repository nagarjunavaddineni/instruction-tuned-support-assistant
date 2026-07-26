from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.common.config import TrainingConfig
from src.training.lm_common import load_model_tokenizer, train

def main():
    c=TrainingConfig.from_env(); model,tok=load_model_tokenizer(c)
    if c.use_4bit: model=prepare_model_for_kbit_training(model)
    model=get_peft_model(model,LoraConfig(r=c.lora_rank,lora_alpha=c.lora_alpha,lora_dropout=c.lora_dropout,target_modules=["q_proj","k_proj","v_proj","o_proj"],bias="none",task_type="CAUSAL_LM"))
    model.print_trainable_parameters(); train(model,tok,c,"outputs/lora")
if __name__=="__main__": main()
