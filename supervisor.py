from openai import OpenAI

def evaluate_and_adjust(
    client, 
    prompt: str, 
    specification: str, 
    output: str, 
    feature_description: str,
    current_steering: float | str, 
    history_context: str = "", 
    supervisor_name: str="gpt-4.1-mini"
    ) -> str:
    
    evaluation_prompt = f"""
    Steering adjusts neural network features to change model behavior. Higher values increase the target behavior but may reduce coherence. Lower values maintain coherence but reduce the desired behavior.

    Original prompt: {prompt}
    Target behavior: {specification}
    Model output: {output}
    Current steering value: {current_steering}
    Steering feature description: {feature_description}
    
    Previous attempts:
    {history_context}
    
    Evaluate if the output achieves the target behavior while remaining coherent and natural.
    
    If satisfactory, respond: stop
    If needs adjustment, respond with a single float between -1.0 and 1.0 for the new steering value.
    If the output is completely off-topic or gibberish, reduce the steering value towards 0.
    """
    
    response = client.chat.completions.create(
        model=supervisor_name,
        messages=[{"role": "user", "content": evaluation_prompt}]
    )
    return response.choices[0].message.content.strip()

def initialize_client():
    return OpenAI()