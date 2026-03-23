import torch

class HybridBackwardWrapper(torch.autograd.Function):
    """
    Custom Autograd Function for Hybrid GPU/CPU execution.
    Forward: Executed on GPU (fast).
    Backward: Re-computes forward on CPU accurately, then performs backward on CPU (stable).
    """
    @staticmethod
    def forward(ctx, func, model_params, *args):
        ctx.func = func
        ctx.model_params = model_params
        
        device = args[0].device if hasattr(args[0], 'device') else torch.device('cpu')
        ctx.device = device
        
        # 1. Fast GPU Forward (No gradients tracked)
        with torch.no_grad():
            gpu_args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
            results = func(*gpu_args)
            
        # 2. Save for backward (CPU)
        ctx.save_for_backward(*[arg.to('cpu') if isinstance(arg, torch.Tensor) else None for arg in args])
        return results

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved_tensors = ctx.saved_variables
        device = ctx.device
        func = ctx.func
        model_params = ctx.model_params
        
        grad_outputs_cpu = [go.to('cpu') if isinstance(go, torch.Tensor) else go for go in grad_outputs]
        
        # 2. Re-enable gradients and run Forward on CPU
        with torch.enable_grad():
            cpu_args = [arg.clone().requires_grad_(True) if arg is not None else None for arg in saved_tensors]
            cpu_results = func(*cpu_args)
            
            if isinstance(cpu_results, tuple):
                valid_results = []
                valid_grads = []
                for res, g in zip(cpu_results, grad_outputs_cpu):
                    if g is not None and isinstance(res, torch.Tensor):
                        valid_results.append(res)
                        valid_grads.append(g)
                torch.autograd.backward(valid_results, valid_grads)
            else:
                torch.autograd.backward(cpu_results, grad_outputs_cpu[0])
                
        # 3. Collect ALL gradients for inputs in *args
        res_grads = [arg.grad.to(device) if (arg is not None and arg.grad is not None) else None for arg in cpu_args]
        
        # 4. Extract parameter gradients specifically for the second return slot ('model_params')
        param_grads = []
        return (None, None, *res_grads)

def hybrid_apply(func, params, *args):
    """
    Signature matches forward: (func, model_params, *args)
    """
    return HybridBackwardWrapper.apply(func, params, *args)
