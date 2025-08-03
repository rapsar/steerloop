# Efficient model steering using supervisor loop

Model steering is the process of nudging the behavior of an LLM by shifting the value of certain features.
Features are linear combinations of neurons (aka latent space directions) corresponding to human-interpretable concepts.

`Goodfire.ai` has developed an [API to access and control different features](https://docs.goodfire.ai/sdk-reference/example) in open-source models. 
The problem is that finding the proper value to steer a model is delicate. 
Too little steering often doesn't change a model's output; too much steering output gibberish.

Here I propose a simple solution where a **supervisor model** (e.g. `gpt-4.1-mini`) evaluates the output of the steerable model compared to the desired behavior, and adjusts the steering value automatically.
The feedback loop converges to the optimal steering value.

Note: Goodfire has a AutoSteer function to find features and values for optimal steering, and it works very well. The supervisor loop is an exploratory alternative.



