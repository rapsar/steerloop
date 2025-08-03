import goodfire
from goodfire import Variant
import os
import supervisor
from openai import OpenAI

def run_steering_loop(
    specification: str,
    prompt: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_iterations: int = 10,
    initial_steering: float = 0.0,
    max_tokens: int = 100
):
    """Run adaptive feature steering with supervisor feedback loop."""
    
    # Initialize clients
    goodfire_client = goodfire.Client(os.getenv("GOODFIRE_API_KEY"))
    openai_client = OpenAI()
    
    # Setup variant and find features
    variant = Variant(model_name)
    features = goodfire_client.features.search(specification, model=variant, top_k=10)
    
    primary_feature = features[0]
    print(f"Using feature: {primary_feature}")
    
    # Initialize tracking
    history = []
    current_steering = initial_steering
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Apply steering
        variant.reset()
        variant.set(primary_feature, current_steering)
        
        # Generate response
        response = goodfire_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=variant,
            max_completion_tokens=max_tokens
        )
        
        output = response.choices[0].message["content"]
        print(f"Steering: {current_steering:.2f}")
        print(f"Output: {output}")
        
        # Build history context
        history_context = _build_history_context(history) if history else ""
        
        # Get supervisor evaluation
        adjust = supervisor.evaluate_and_adjust(
            client=openai_client,
            prompt=prompt,
            specification=specification,
            output=output,
            current_steering=current_steering,
            feature_description=str(primary_feature),
            history_context=history_context
        )
        
        print(f"Supervisor: {adjust}")
        
        # Record iteration
        history.append({
            'iteration': iteration + 1,
            'steering': current_steering,
            'output': output,
            'suggestion': adjust.strip()
        })
        
        # Check termination
        if adjust.strip().lower() == "stop":
            print(f"\n✓ Supervisor satisfied after {iteration + 1} iterations")
            break
        
        # Update steering
        try:
            new_steering = float(adjust.strip())
            # Clamp to reasonable bounds
            current_steering = max(-2.0, min(2.0, new_steering))
        except ValueError:
            print(f"✗ Invalid steering value: {adjust}")
            break
    
    if iteration == max_iterations - 1:
        print(f"\n⚠ Reached max iterations ({max_iterations})")
    
    # Save results
    results = {
        'session_info': {
            'prompt': prompt,
            'specification': specification,
            'model': model_name,
            'supervisor': 'gpt-4.1-mini',
            'feature': str(primary_feature),
            'total_iterations': len(history),
            'final_steering': current_steering,
            'converged': adjust.strip().lower() == "stop"
        },
        'iterations': history
    }
    
    # Save to file
    save_session_to_file(results)
    
    return results

def save_session_to_file(results):
    """Save complete session to text file."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversations/steering_session_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        # Header
        f.write("=" * 60 + "\n")
        f.write("ADAPTIVE FEATURE STEERING SESSION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Prompt: {results['session_info']['prompt']}\n")
        f.write(f"Specification: {results['session_info']['specification']}\n")
        f.write(f"Model: {results['session_info']['model']}\n")
        f.write(f"Supervisor: {results['session_info']['supervisor']}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Feature: {results['session_info']['feature']}\n")
        f.write("=" * 60 + "\n\n")
        
        # Iterations
        for iter_data in results['iterations']:
            f.write(f"--- Iteration {iter_data['iteration']} ---\n")
            f.write(f"Steering: {iter_data['steering']:.2f}\n")
            f.write(f"Output: {iter_data['output']}\n")
            f.write(f"Supervisor: {iter_data['suggestion']}\n\n")
        
        # Summary
        f.write("=" * 60 + "\n")
        f.write("SESSION SUMMARY\n")
        f.write("=" * 60 + "\n")
        for key, value in results['session_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    print(f"\nSession saved to: {filename}")
    
    return results

def _build_history_context(history):
    """Build formatted history context for supervisor."""
    return "\n".join([
        f"Iteration {h['iteration']}: Steering={h['steering']:.2f}, Suggestion={h['suggestion']}"
        for h in history
    ])


if __name__ == "__main__":
    result = run_steering_loop(
        specification="Connecticut",
        prompt="Which US state should I visit this summer? Just respond in one word.",
        max_iterations=10
    )
    
    print(f"\nCompleted with {result['session_info']['total_iterations']} iterations")
    print(f"Final steering: {result['session_info']['final_steering']:.2f}")