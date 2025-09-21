import json
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import glob
from collections import defaultdict

def extract_all_image_features():
    """高效方案：先提取所有图片特征，再按item_map整理"""
    
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # 数据根目录
    data_root = "/mnt/asaliao/cafidata/IQON3000"
    
    # 第一步：收集所有唯一的图片文件
    print("扫描所有图片文件...")
    all_image_files = glob.glob(os.path.join(data_root, "*", "*", "*_m.jpg"))
    print(f"找到 {len(all_image_files)} 个图片文件")
    
    # 提取item_id到文件路径的映射（去重）
    item_to_paths = defaultdict(list)
    for file_path in all_image_files:
        # 从路径中提取item_id
        filename = os.path.basename(file_path)
        item_id = filename.replace('_m.jpg', '')
        item_to_paths[item_id].append(file_path)
    
    print(f"找到 {len(item_to_paths)} 个唯一产品")
    
    # 第二步：批量提取所有唯一图片的特征
    print("开始提取所有唯一图片的特征...")
    
    # 存储所有特征 {item_id: feature}
    all_features_dict = {}
    processed_count = 0
    error_count = 0
    
    # 分批处理以避免内存问题
    batch_size = 256
    item_ids = list(item_to_paths.keys())
    
    for i in tqdm(range(0, len(item_ids), batch_size), desc="提取特征"):
        batch_item_ids = item_ids[i:i + batch_size]
        batch_images = []
        valid_indices = []  # 记录哪些索引的图片可以正常处理
        
        for item_id in batch_item_ids:
            # 取第一个可用的路径
            image_path = item_to_paths[item_id][0]
            try:
                image = Image.open(image_path).convert('RGB')
                image_input = preprocess(image)
                batch_images.append(image_input)
                valid_indices.append(item_id)
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"图片读取错误: {item_id}, {e}")
                continue
        
        if not batch_images:
            continue
            
        # 批量处理图片
        try:
            image_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_features = model.encode_image(image_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            # 存储特征到字典
            for j, item_id in enumerate(valid_indices):
                all_features_dict[item_id] = batch_features[j].cpu()
                processed_count += 1
                
        except Exception as e:
            print(f"批量处理错误: {e}")
            # 如果批量处理失败，回退到逐个处理
            for j, item_id in enumerate(valid_indices):
                try:
                    image_path = item_to_paths[item_id][0]
                    image = Image.open(image_path).convert('RGB')
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feature = model.encode_image(image_input)
                        feature = feature / feature.norm(dim=-1, keepdim=True)
                    
                    all_features_dict[item_id] = feature.cpu().squeeze(0)
                    processed_count += 1
                except Exception as e2:
                    error_count += 1
                    if error_count <= 5:
                        print(f"回退处理错误: {item_id}, {e2}")
    
    print(f"成功提取 {processed_count} 个产品的特征")
    print(f"处理错误: {error_count} 个产品")
    
    # 保存原始特征字典
    torch.save(all_features_dict, 'clip_features_dict.pt')
    print(f"原始特征字典已保存到 clip_features_dict.pt")
    
    return all_features_dict

def create_final_tensor(all_features_dict):
    """根据item_map创建最终的特征张量"""
    
    # 加载item_map.json
    with open('/home/asaliao/APCL/dataset/IQON3000/data/item_map.json', 'r') as f:
        item_map = json.load(f)
    
    num_items = len(item_map)
    feature_dim = 512
    
    # 初始化最终的特征张量
    final_features = torch.zeros(num_items, feature_dim)
    missing_items = []
    
    print(f"根据item_map整理 {num_items} 个产品的特征...")
    
    # 按照item_map的顺序填充特征张量
    for item_id, idx in tqdm(item_map.items(), desc="整理特征"):
        if item_id in all_features_dict:
            final_features[idx] = all_features_dict[item_id]
        else:
            missing_items.append(item_id)
            if len(missing_items) <= 10:
                print(f"警告: 产品 {item_id} 在特征字典中不存在")
    
    print(f"整理完成! 缺失 {len(missing_items)} 个产品")
    
    # 保存最终的特征张量
    torch.save(final_features, '/home/asaliao/APCL/dataset/IQON3000/feat/clip_vis_features_indexedtensor')
    print(f"最终特征张量已保存到 clip_vis_features_indexedtensor")
    print(f"特征张量形状: {final_features.shape}")
    
    # 保存缺失产品列表
    with open('missing_items.json', 'w') as f:
        json.dump(missing_items, f, indent=2)
    
    return final_features

def verify_results():
    """验证结果"""
    print("\n=== 结果验证 ===")
    
    # 验证最终特征张量
    try:
        final_features = torch.load('/home/asaliao/APCL/dataset/IQON3000/feat/clip_vis_features_indexedtensor')
        print(f"最终特征张量形状: {final_features.shape}")
        
        # 验证特征质量
        norms = torch.norm(final_features, dim=1)
        zero_features = (norms == 0).sum().item()
        valid_features = final_features.shape[0] - zero_features
        
        print(f"有效特征数量: {valid_features}")
        print(f"零特征数量: {zero_features}")
        print(f"特征范数范围: [{norms.min().item():.4f}, {norms.max().item():.4f}]")
        
    except FileNotFoundError:
        print("最终特征文件不存在")
    
    # 验证原始特征字典
    try:
        features_dict = torch.load('clip_features_dict.pt')
        print(f"原始特征字典大小: {len(features_dict)}")
    except FileNotFoundError:
        print("原始特征字典不存在")

if __name__ == "__main__":
    print("=== 高效特征提取方案 ===")
    print("步骤1: 提取所有唯一图片的特征")
    print("步骤2: 根据item_map整理成最终张量")
    
    # 第一步：提取所有特征
    all_features_dict = extract_all_image_features()
    
    # 第二步：整理成最终张量
    final_features = create_final_tensor(all_features_dict)
    
    # 验证结果
    verify_results()
    
    print("\n=== 使用说明 ===")
    print("1. clip_vis_features_indexedtensor: 包含所有找到的特征 {item_id: feature}")
    print("2. clip_vis_features_indexedtensor: 按item_map顺序排列的特征张量 [num_items, 512]")
    print("3. missing_items.json: 缺失的产品列表")
    print("4. 可以直接使用: features = torch.load('clip_vis_features_indexedtensor')")