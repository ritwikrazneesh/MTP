"""OpenCLIP adapter as fallback."""

import torch
import torch.nn as nn
import open_clip


class OpenCLIPAdapter(nn.Module):
    """
    OpenCLIP adapter for image classification.
    Fallback option when RemoteCLIP is not available.
    """
    
    def __init__(self, class_names, device="cuda", model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        self.class_names = class_names
        self.device = device
        
        # Load OpenCLIP model
        print(f"Loading OpenCLIP model ({model_name})...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
        except Exception as e:
            print(f"Warning: Could not download pretrained weights: {e}")
            print("Loading model without pretrained weights (random initialization)...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None
            )
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Create text features for class names
        self._create_text_features()
        
        # Initialize learnable prompt context
        self._init_prompt_context()
    
    def _create_text_features(self):
        """Create text embeddings for class names."""
        # Create prompts with template
        prompts = [f"a photo of {name.lower()}" for name in self.class_names]
        text_tokens = self.tokenizer(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.register_buffer("text_features", text_features)
    
    def _init_prompt_context(self):
        """Initialize learnable prompt context for O-TPT."""
        # Get embedding dimension from the visual encoder
        # For ViT-B-32, the embedding dimension is typically 512
        try:
            if hasattr(self.model, 'text_projection'):
                embed_dim = self.model.text_projection.shape[1]
            elif hasattr(self.model, 'transformer'):
                embed_dim = self.model.transformer.width
            else:
                # Default for ViT-B-32
                embed_dim = 512
        except:
            embed_dim = 512
        
        # Initialize learnable context vectors (4 context tokens)
        ctx_init = torch.empty(4, embed_dim)
        nn.init.normal_(ctx_init, std=0.02)
        self.prompt_ctx = nn.Parameter(ctx_init)
    
    def reset_prompt_context(self):
        """Reset prompt context to initial values."""
        nn.init.normal_(self.prompt_ctx.data, std=0.02)
    
    def encode_image(self, images):
        """Encode images to features."""
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def forward(self, images):
        """
        Forward pass with learnable prompts.
        
        Args:
            images: Input images [B, 3, 224, 224]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Encode images
        image_features = self.encode_image(images)
        
        # Use fixed text features for now (O-TPT will optimize prompt_ctx)
        # In full O-TPT, we would incorporate prompt_ctx into text encoding
        logits = 100.0 * image_features @ self.text_features.T
        
        return logits
