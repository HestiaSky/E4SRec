import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']

        print(f'Initializing language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(self.args['base_model'], load_in_8bit=True,
                                                            torch_dtype=torch.float16, device_map=self.args['device_map'])
        self.llama_model = prepare_model_for_int8_training(self.llama_model)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.args['base_model'], use_fast=False)
        self.llama_tokenizer.pad_token = 0
        self.llama_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llama_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False, add_special_tokens=False)
        self.response_ids, self.response_mask = self.llama_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False, add_special_tokens=False)
        (self.instruct_ids, self.instruct_mask,
         self.response_ids, self.response_mask) = (self.instruct_ids.cuda(), self.instruct_mask.cuda(),
                                                   self.response_ids.cuda(), self.response_mask.cuda())
        print('Language decoder initialized.')

        self.user_embeds = nn.Embedding.from_pretrained(self.args['user_embeds'], freeze=True)
        self.user_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True)
        self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False)
        self.post_init()

    def predict(self, inputs):
        instruct_embeds = self.llama_model.model.model.embed_tokens(self.instruct_ids)
        response_embeds = self.llama_model.model.model.embed_tokens(self.response_ids)

        if self.user_embeds.shape[0] > 1:
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([self.instruct_mask, torch.ones(inputs.shape[1]).cuda(), self.response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(input_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.hidden_states[:, -1]
        pooled_logits = self.score(pooled_output)

        return pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask, labels):
        bs = inputs.shape[0]
        instruct_embeds = self.llama_model.model.model.embed_tokens(self.instruct_ids).expand(bs, -1, -1)
        response_embeds = self.llama_model.model.model.embed_tokens(self.response_ids).expand(bs, -1, -1)

        if self.user_embeds.shape[0] > 1:
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([self.instruct_mask, inputs_mask, self.response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(input_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.hidden_states[:, -1]
        pooled_logits = self.score(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.output_dim), labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )










