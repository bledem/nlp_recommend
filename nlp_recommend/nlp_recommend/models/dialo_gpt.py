from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Conversation
import torch



class ConvAI:
    def __init__(self):
        self.model_name = 'microsoft/DialoGPT-large'
        self.chat_history_ids = None
        self.prev_ans = None

    def load_model(self):
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get(self, user_input, it=0):
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        # append the new user input tokens to the chat history
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) 
        else:
            bot_input_ids = new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens, 
        self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        # print(self.chat_history_ids.shape,  bot_input_ids.shape)
        ans = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # Transforming affirmation into question when model fails
        if ans == self.prev_ans or ans == user_input:
            ans = ans.replace('I', 'You')
            ans = ans.replace('am', 'are')
            ans = ans.replace('We', 'You')
            ans = ans.replace('.', '?')
            if '?' not in ans:
                ans = ans + '?'
        self.prev_ans = ans
        return ans

if __name__ == '__main__':
    model = ConvAI()
    user_sentence = [
        'I want to read some adventures. Do you know where should I go?',
    'Tell me what I can learn from you',
     'I am afraid of everything. How can I cure myself?',
     'I want to learn how to be stronger. How can I be stronger?'
     ]
    model.load_model()
    ans = model.get('Hi! How are you?')
    for sentence in user_sentence:
        print(f' >> user:    {sentence}')
        print(f' >> ans: {model.get(sentence)}')


# Possible sentence
## Hey 