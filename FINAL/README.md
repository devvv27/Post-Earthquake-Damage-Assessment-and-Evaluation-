| **Model** | **Loss** | **IoU** | **mIoU** | **Dice Coefficient** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|------------|----------|---------|-----------|----------------------|---------------|----------------|-------------|---------------|
| **Siamese U-Net with CBAM (spatial and channel attention)** | 0.374489 | 0.4464 | 0.6995 | 0.6172 | 0.9543 | 0.6137 | 0.6208 | 0.6172 |
| **Siamese U-Net with CBAM (spatial attention)** | 0.379060 | 0.4360 | 0.6951 | 0.6073 | 0.9558 | 0.6424 | 0.5758 | 0.6073 |
| **Siamese U-Net with CBAM (channel attention)** | 0.373318 | 0.4569 | 0.7038 | 0.6273 | 0.9527 | 0.5889 | 0.6710 | 0.6273 |
| **Siamese U-Net with Coordinate Attention(Spatial and channel)** | 0.371130 | 0.4564 | 0.7054 | 0.6268 | 0.9561 | 0.6321 | 0.6215 | 0.6268 |
| **Siamese U-Net with Coordinate spatial attention** | 0.372929 | 0.4467 | 0.6996 | 0.6175 | 0.9544 | 0.6146 | 0.6204 | 0.6175 |
| **Siamese U-Net with Coordinate channel attention** | 0.368599 | 0.4592 | 0.7074 | 0.6293 | 0.9572 | 0.6479 | 0.6118 | 0.6293 |
| **Siamese U-Net with ECA** | 0.368067 | 0.4693 | 0.7141 | 0.6388 | 0.9604 | 0.6959 | 0.5903 | 0.6388 |
| **Siamese U-Net with Multi - Scale ECA** | 0.371324 | 0.4696 | 0.7120 | 0.6390 | 0.9562 | 0.6250 | 0.6538 | 0.6390 |
| **Siamese U-Net with SE with r = 16** | 0.367700 | 0.4665 | 0.7116 | 0.6362 | 0.9583 | 0.6594 | 0.6146 | 0.6362 |
| **Siamese U-Net with SE with r = 12** | 0.362286 | 0.4620 | 0.7091 | 0.6320 | 0.9578 | 0.6545 | 0.6110 | 0.6320 |
| **Siamese U-Net with SE with r = 8** | 0.367155 | 0.4584 | 0.7062 | 0.6287 | 0.9556 | 0.6249 | 0.6324 | 0.6287 |
| **Siamese U-Net with local Cross Attention ** | 0.384571 | 0.4278 | 0.6908 | 0.5993 | 0.9553 | 0.6402 | 0.5633 | 0.5993 |
| **Siamese U-Net with local Cross Attention at first 2 levels and global cross attention in next 3 ** | 0.395333 | 0.3776 | 0.6627 | 0.5482 | 0.9494 | 0.5828 | 0.5175 | 0.5482 |
| **Siamese U-Net with local Cross Attention at each level with eca ** | 0.390024 | 0.3951 | 0.6747 | 0.5664 | 0.9556 | 0.6742 | 0.4883 | 0.5664 |
| **Siamese U-Net with local Cross Attention at first 2 levels and global cross attention in next 3 with eca ** | 0.390205 | 0.3884 | 0.6636 | 0.5595 | 0.9410 | 0.5024 | 0.6312 | 0.5595 |
| **Siamese U-Net with SK** |  |  |  |  |  |  |  |  |
| **Siamese U-Net with SRM** |  |  |  |  |  |  |  |  |
| **Siamese U-Net with Triplet Attention ** |  |  |  |  |  |  |  |  |
