"""
Google Colab demo for PromptSage
"""

def setup_promptsage():
    """Set up the PromptSage package in Google Colab."""
    import os
    import sys
    
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if not IN_COLAB:
        print("This script is designed to run in Google Colab.")
        return False
    
    # Install required packages
    !pip install -q transformers datasets torch matplotlib seaborn wordcloud nltk

    # Clone the repository (assuming it's on GitHub at this point)
    !git clone https://github.com/fardeen0424/PromptSage.git
    
    # Add to Python path
    promptsage_path = os.path.join(os.getcwd(), "promptsage")
    if os.path.exists(promptsage_path):
        sys.path.append(promptsage_path)
        print("PromptSage successfully set up!")
        return True
    else:
        print("Failed to set up PromptSage.")
        return False

def download_model():
    """Download a model for use with PromptSage."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("Downloading model (this may take a while)...")
    model_name = "EleutherAI/gpt-neo-1.3B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"Model {model_name} downloaded successfully!")
    return model_name, model, tokenizer

def optimize_prompt_demo(prompt, model_name, strategy="auto"):
    """Run a prompt optimization demo."""
    from promptsage import PromptOptimizer
    import time
    
    print(f"Optimizing prompt using {strategy} strategy...")
    start_time = time.time()
    
    optimizer = PromptOptimizer(
        model_name=model_name,
        optimization_strategy=strategy
    )
    
    optimized_prompt, metrics = optimizer.optimize(
        prompt=prompt, 
        num_iterations=3,
        verbose=True
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n=== Results ===")
    print(f"Original prompt: {prompt}")
    print(f"Optimized prompt: {optimized_prompt}")
    print(f"Optimization time: {elapsed_time:.2f} seconds")
    
    # Show improvement metrics
    if "improvement" in metrics:
        print("\nImprovement Metrics:")
        for metric, value in metrics["improvement"].items():
            print(f"  {metric}: {value:.4f}")
    
    # Compare responses
    comparison = optimizer.compare(prompt, optimized_prompt, visualize=True)
    
    print("\n=== Responses ===")
    print(f"Original response: {comparison['original_response']}")
    print(f"Optimized response: {comparison['optimized_response']}")
    
    # Display visualizations
    if "visualization" in comparison:
        from IPython.display import display, Image
        import base64
        import io
        
        for viz_name, viz_data in comparison["visualization"].items():
            if viz_data["type"] == "image":
                img_bytes = base64.b64decode(viz_data["data"])
                display(Image(data=img_bytes))
    
    return optimized_prompt, metrics

def run_demo():
    """Run the full PromptSage demo in Colab."""
    if not setup_promptsage():
        return
        
    # Download model
    model_name, _, _ = download_model()
    
    # Demo prompt
    demo_prompt = "Explain quantum computing."
    
    # Run optimization demo
    optimize_prompt_demo(demo_prompt, model_name)
    
    # Allow user input
    from IPython.display import clear_output
    
    while True:
        user_prompt = input("Enter a prompt to optimize (or 'q' to quit): ")
        if user_prompt.lower() == 'q':
            break
            
        strategy = input("Enter optimization strategy (evolution, meta, contrastive, auto): ")
        if not strategy or strategy not in ["evolution", "meta", "contrastive", "auto"]:
            strategy = "auto"
            
        clear_output()
        optimize_prompt_demo(user_prompt, model_name, strategy)

if __name__ == "__main__":
    run_demo()