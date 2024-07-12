

chars = ""
with open("data/vocab.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        
    
# the tokenizer 
def tokenizer( input , mode):

    
    if mode=="encoder":
        string_to_int = { ch:i for i,ch in enumerate(chars) }
        encode = lambda s: [string_to_int[c] for c in s]
        return encode(input)
        
    if mode=="decoder":
        int_to_string = { i:ch for i,ch in enumerate(chars) }
        decode = lambda l: ''.join([int_to_string[i] for i in l])
        return decode(input)
        
        
