# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

# Import model components and graph construction functions from your existing modules
from model_components import EfficientNetV2FeatureExtractor, GATGNN, TransformerEncoder, MLPBlock
from graph_construction import build_graph_from_patches, build_graph_data_from_patches

# Import the LISA dataset loader (ensure this module is in your PYTHONPATH)
from lisa import LISA

# Set the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# %%

class SAGViTLISAClassifier(nn.Module):
    """
    SAG‑ViT re‐implemented for the LISA dataset.

    This model:
      - Uses a CNN backbone (EfficientNetV2‑S) to extract high-fidelity feature maps.
      - Splits the feature maps into non-overlapping patches (default patch size is (4,4)).
      - Constructs a graph over the patches based on spatial adjacency and refines patch embeddings with a GAT.
      - Processes an extra token along with the patch embedding using a Transformer encoder to capture global dependencies.
      - Classifies the aggregated features via an MLP head.

    **Note:** We assume the CNN backbone outputs feature maps with 160 channels so that a (4×4) patch yields a flattened vector of
    160×4×4 = 2560 dimensions.
    """
    def __init__(
        self,
        patch_size=(4, 4),
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        gcn_hidden=128,
        gcn_out=64,
        backbone_pretrained=True,
        unfreeze_blocks=6,
    ):
        super(SAGViTLISAClassifier, self).__init__()
        # CNN backbone for feature extraction
        self.cnn = EfficientNetV2FeatureExtractor(pretrained=backbone_pretrained, unfreeze_blocks=unfreeze_blocks)
        
        # Assume the CNN outputs 160 channels
        self.backbone_out_channels = 160  
        patch_h, patch_w = patch_size
        # Compute the patch vector dimension: channels * patch_height * patch_width
        patch_vector_dim = self.backbone_out_channels * patch_h * patch_w
        
        # Graph Attention Network to refine patch embeddings
        self.gcn = GATGNN(
            in_channels=patch_vector_dim,
            hidden_channels=gcn_hidden,
            out_channels=gcn_out,
        )
        
        # Positional embedding for the patch token and an extra learnable token (like a [CLS] token)
        # For simplicity, we assume that d_model == gcn_out
        self.positional_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.extra_embedding = nn.Parameter(torch.randn(1, d_model))
        
        # Transformer encoder to capture global dependencies
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
        
        # MLP classification head
        self.mlp = MLPBlock(d_model, d_model, num_classes)
        
        self.patch_size = patch_size

    def forward(self, x):
        # 1. Extract high-fidelity feature maps from the CNN backbone.
        feature_map = self.cnn(x)  # Expected shape: (B, C, H', W')
        
        # 2. Partition the feature map into patches and build a spatial graph.
        G_global_batch, patches = build_graph_from_patches(feature_map, self.patch_size)
        data_list = build_graph_data_from_patches(G_global_batch, patches)
        batch = Batch.from_data_list(data_list).to(x.device)
        
        # 3. Apply the GAT (graph convolution) to refine patch embeddings.
        x_gcn = self.gcn(batch)  # Shape: (B, gcn_out)
        
        # 4. Prepare sequence for the Transformer encoder:
        #    - Add positional embedding to the graph-level patch embedding.
        patch_embeddings = x_gcn.unsqueeze(1) + self.positional_embedding  # (B, 1, gcn_out)
        
        # Concatenate an extra learnable token (like a [CLS] token)
        B = x.size(0)
        extra_token = self.extra_embedding.unsqueeze(0).expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([patch_embeddings, extra_token], dim=1)  # (B, 2, d_model)
        
        # 5. Process tokens with the Transformer encoder.
        x_trans = self.transformer_encoder(tokens)  # (B, 2, d_model)
        # Global pooling over tokens (here, simply a mean)
        x_pooled = x_trans.mean(dim=1)  # (B, d_model)
        
        # 6. Classification via the MLP head.
        out = self.mlp(x_pooled)
        return out




