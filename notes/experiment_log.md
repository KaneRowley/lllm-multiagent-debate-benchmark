30 October - First, I installed Ollama and downloaded Qwen3-4B, a LLM model that is capable of maths, coding and other general work. I connected my python program to Ollama through its API (which is Open-AI compatitable). I use client.chat.completions.create() to initiate a request to the model (formatted by OpenAI client library) and returns the model's generated response. Used localhost rather than Hugging Face URL so I can inference locally without relying on remote model-hosting service. .env file is used to store config values separately from my source code.

1 November - Using codex, I created a simple calculator tool function that defines basic arithmetic operations. Important to note - the LLM doesn't directly execute the function, it makes a decision it needs a calculation and generates a tool call ("calculate", "23", "17", "multiply") requesting python to execute it. The JSON schema helps the model produce a correctly structured request. This result is then returned to the LLM for it to make its final response. Note that we do not print the result of that python outputs, but we send a tool-result message back to the model so it can: a) Incorporate calculation into answer b) Continue reasoning or request another tool if necessary c) Produce a coherent final response for user. 

Ask prompt -> LLM thinks -> it returns a tool-call request -> Python interprets request and dispatches to appropriate function -> Performs actual calculation -> Tool-result message returned back to LLM -> LLM uses result to think

Motivation of tool-use implementation

I first let a single Qwen3 have a go at numerous arithmetic problems (in terminal, not on hello_model.py. In particular, I asked:
"A shop sells 17 boxes of pencils. "
                    "Each box contains 24 pencils. "
                    "How many pencils are there altogether?"

A simple worded multiplication question and it reached the correct solution of 408. What was interesting wasn't the actual solution but it's working out:

Hmm, 17×24... I should calculate this carefully. Let me break it down. One way is to use the 
distributive property: 17×24 = 17×(20+4) = (17×20) + (17×4). 

17×20 is 340, and 17×4 is 68. Adding those gives 340+68=408. 

Rather than "mentally" working what 17 x 24, it simplifies the problem by using the distributive property. While this does reach the same answer (although the model makes sure it is correct by using alternative methods) I thought why not provide Qwen with a calculator? When faced with similar or harder problems, Qwen will constantly check each arithmetic it will do and be stuck when it calculates the wrong answer. By providing a calculator, we can provide strong evidence that it is correct which may increase its accuracy and computation time.

Hypothesis

This raises multiple questions:

1)Could a calculator produce a more reliable arithmetic result than relying on the LLM's generated answer?
2)Would the LLM recognise when using the calculator is worthwhile, and invoke it appropriately?
3)Does using a calculator actually reduce computation time, or does the extra model interaction outweigh the time saved? 
4)Does the tool increase the model's accuracy?
5)Can unessecary tool calls make model's performance worse?

My hypothesis is that the calculator tool gives strong evidence to Qwen, but it may be overreliant on it causing it to think less itself. I believe computational time will be saved for simple arithmetic problems, but it may prolong problems that require multiple of these.

I will try to send more problems for Qwen to solve, using and without using a calculator and compare its performance.

November 3 - I wanted to prepare a small development dataset so I can test my model on mathematical questions. I used the GSM8K benchmark dataset and loaded its training split and selected 30 development questions randomly. Added a ground-truth extraction function which extracts the numerical answer after ####. I used Codex to help me add stricter answer validation. I will load test split when evaluating models. Plan to run these questions on a single model with/without a tool.
