"""O-TPT (Online Test-Time Prompt Tuning) core implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


class OTPT:
    """
    O-TPT: Online Test-Time Prompt Tuning.
    
    Performs entropy minimization on confident samples with orthogonality
    regularization, using AMP and per-batch prompt context reset.
    """
    
    def __init__(self, model, lr=0.001, steps=1, entropy_threshold=0.6, 
                 orth_reg=0.1, device="cuda"):
        """
        Initialize O-TPT.
        
        Args:
            model: Model with prompt_ctx parameter
            lr: Learning rate for prompt tuning
            steps: Number of optimization steps per batch
            entropy_threshold: Confidence threshold for filtering samples
            orth_reg: Orthogonality regularization weight
            device: Device to run on
        """
        self.model = model
        self.lr = lr
        self.steps = steps
        self.entropy_threshold = entropy_threshold
        self.orth_reg = orth_reg
        self.device = device
        
        # Setup optimizer for prompt context
        self.optimizer = torch.optim.AdamW([self.model.prompt_ctx], lr=lr)
        
        # Setup AMP
        self.scaler = GradScaler()
    
    def entropy_loss(self, logits):
        """
        Compute entropy loss for confident samples.
        
        Args:
            logits: Model logits [B, num_classes]
            
        Returns:
            loss: Entropy loss
        """
        probs = F.softmax(logits, dim=1)
        
        # Compute confidence (max probability)
        confidence, _ = probs.max(dim=1)
        
        # Filter confident samples
        mask = confidence > self.entropy_threshold
        
        if mask.sum() == 0:
            # No confident samples, return zero loss
            return torch.tensor(0.0, device=self.device)
        
        # Compute entropy for confident samples
        probs_confident = probs[mask]
        entropy = -(probs_confident * torch.log(probs_confident + 1e-10)).sum(dim=1)
        
        return entropy.mean()
    
    def orthogonality_loss(self):
        """
        Compute orthogonality regularization loss for prompt context.
        
        Returns:
            loss: Orthogonality loss
        """
        # Get prompt context vectors
        ctx = self.model.prompt_ctx
        
        # Compute Gram matrix
        gram = ctx @ ctx.T
        
        # Orthogonality: minimize off-diagonal elements
        n = ctx.size(0)
        identity = torch.eye(n, device=self.device)
        
        # L2 distance from identity
        orth_loss = ((gram - identity) ** 2).sum() / (n * (n - 1))
        
        return orth_loss
    
    def forward(self, images):
        """
        Forward pass with O-TPT optimization.
        
        Args:
            images: Input images [B, 3, 224, 224]
            
        Returns:
            probs: Class probabilities [B, num_classes]
        """
        # Reset prompt context for this batch
        self.model.reset_prompt_context()
        
        # Optimize prompt context for this batch
        for step in range(self.steps):
            self.optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                logits = self.model(images)
                
                # Compute losses
                ent_loss = self.entropy_loss(logits)
                orth_loss = self.orthogonality_loss()
                
                # Total loss
                loss = ent_loss + self.orth_reg * orth_loss
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # Final forward pass for inference
        with torch.no_grad():
            with autocast():
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
        
        return probs
