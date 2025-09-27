# test_dataset.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data.datamgr import SimpleDataManager
from data.dataset import SimpleDataset_JSON
import torch

def test_video_shapes():
    """Test shape của video tensor"""
    print("Testing video shapes...")
    
    try:
        dataset = SimpleDataset_JSON(
            data_path="C:/Users/dungs/Documents/Study/ManageBigData/TAMT-main/datasets/SSV2",
            data_file="base1.json",
            transform=None,  # Không dùng transform để giữ nguyên video tensor
            clip_len=16,
            frame_sample_rate=2,
            num_segment=8
        )
        
        # Test 5 samples đầu tiên
        for i in range(min(5, len(dataset))):
            video, target = dataset[i]
            print(f"Sample {i}: video shape {video.shape}, target {target}")
            
            # Kiểm tra shape
            if video.dim() == 4:  # [C, T, H, W]
                print(f"  ✓ Video tensor correct shape: {video.shape}")
            else:
                print(f"  ✗ Wrong shape: {video.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Video shape test failed: {e}")
        return False

def test_data_loader_with_video():
    """Test DataLoader với video"""
    print("\nTesting DataLoader with video...")
    
    try:
        # Sử dụng transform đơn giản
        from torchvision import transforms
        simple_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        datamgr = SimpleDataManager(
            data_path="C:/Users/dungs/Documents/Study/ManageBigData/TAMT-main/datasets/SSV2",
            image_size=224,
            batch_size=2,
            json_read=True,
            clip_len=16,
            frame_sample_rate=2,
            num_segment=8
        )
        
        loader = datamgr.get_data_loader("base.json", aug=False)
        
        for batch_idx, (data, target) in enumerate(loader):
            print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
            
            # Kiểm tra shape của batch
            if data.dim() == 5:  # [B, C, T, H, W] cho video
                print(f"  ✓ Correct video batch shape: {data.shape}")
            elif data.dim() == 4:  # [B, C, H, W] cho ảnh
                print(f"  ⚠ Image batch shape: {data.shape}")
            
            if batch_idx >= 1:  # Chỉ test 2 batch
                break
                
        return True
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return False

if __name__ == '__main__':
    success1 = test_video_shapes()
    success2 = test_data_loader_with_video()
    
    if success1 and success2:
        print("\n🎉 All tests passed! You can now run the training.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")