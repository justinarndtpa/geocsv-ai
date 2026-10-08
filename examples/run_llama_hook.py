import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from geocsv.core.cayley import ExactCayleyRetraction

def run_stiefel_hook_experiment(model_path: str, prompt: str):
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Llama-3.2-1B hidden dimension is 2048
    d = model.config.hidden_size
    r = 16 # Low-rank generator dimension
    
    # Generate random orthonormal bases for A and B to create a Stiefel manifold constraint
    # In a real scenario, A and B would be learned steering vectors (e.g. from an SAE)
    # Using float32 for inverse calculation compatibility
    A = torch.randn(d, r, device=model.device, dtype=torch.float32)
    A, _ = torch.linalg.qr(A) # Orthogonalize
    
    B = torch.randn(d, r, device=model.device, dtype=torch.float32)
    B, _ = torch.linalg.qr(B) # Orthogonalize
    
    # Initialize the Cayley Retraction engine with float32
    cayley = ExactCayleyRetraction(dimension=d, rank=r, eta=0.01).to(model.device).to(torch.float32)
    
    # Precompute the Stiefel Projection Matrix R in SO(d) (computed in float32)
    print("Computing Symplectic Stiefel Matrix (R) via Cayley Retraction...")
    R_float32 = cayley(A, B)
    
    # Cast back to bfloat16 for inference
    R = R_float32.to(torch.bfloat16)
    
    # Define the PyTorch Forward Hook
    def stiefel_manifold_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            projected_states = torch.matmul(hidden_states, R.T)
            return (projected_states,) + output[1:]
        else:
            hidden_states = output
            projected_states = torch.matmul(hidden_states, R.T)
            return projected_states

    # Register the hook on a middle layer (e.g., layer 8 of 16 for 1B)
    layer_idx = model.config.num_hidden_layers // 2
    print(f"Attaching Stiefel projection hook to layer {layer_idx}...")
    hook_handle = model.model.layers[layer_idx].register_forward_hook(stiefel_manifold_hook)
    
    # Run Generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\nPrompt: {prompt}")
    print("Generating response with continuous Stiefel steering...\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # Cleanup hook
    hook_handle.remove()

if __name__ == "__main__":
    MODEL_DIR = r"C:\BlueDot\Llama-3.2-1B"
    TEST_PROMPT = "The fundamental difference between empirical machine learning and formal verification is"
    
    run_stiefel_hook_experiment(MODEL_DIR, TEST_PROMPT)
