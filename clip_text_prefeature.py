import json
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import glob
from collections import defaultdict
import re

def extract_text_features():
    """提取所有产品的文字特征并按item_map顺序保存"""
    
    # 加载CLIP模型
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # 数据根目录
    data_root = "/mnt/asaliao/cafidata/IQON3000"
    
    # 加载item_map.json
    with open('APCL/dataset/IQON3000/data/item_map.json', 'r') as f:
        item_map = json.load(f)
    
    num_items = len(item_map)
    feature_dim = 512
    print(f"总共需要处理 {num_items} 个产品，特征维度: {feature_dim}")
    
    # 第一步：收集所有JSON文件
    print("扫描所有JSON文件...")
    all_json_files = glob.glob(os.path.join(data_root, "*", "*", "*.json"))
    print(f"找到 {len(all_json_files)} 个JSON文件")
    
    # 第二步：构建item_id到文本内容的映射
    print("解析JSON文件，提取文本内容...")
    item_to_text = {}
    processed_jsons = 0
    error_jsons = 0
    
    for json_file in tqdm(all_json_files, desc="解析JSON"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取文本信息（根据实际JSON结构调整）
            text_content = extract_text_from_json(data)
            
            # 获取文件夹中的所有图片ID
            folder_path = os.path.dirname(json_file)
            image_files = glob.glob(os.path.join(folder_path, "*_m.jpg"))
            
            for img_file in image_files:
                item_id = os.path.basename(img_file).replace('_m.jpg', '')
                if item_id not in item_to_text and text_content:  # 避免覆盖和空文本
                    item_to_text[item_id] = text_content
            
            processed_jsons += 1
            
        except Exception as e:
            error_jsons += 1
            if error_jsons <= 5:
                print(f"JSON解析错误: {json_file}, 错误: {str(e)}")
    
    print(f"成功解析 {processed_jsons} 个JSON文件，错误: {error_jsons}")
    print(f"收集到 {len(item_to_text)} 个产品的文本信息")
    
    # 第三步：提取文本特征
    print("开始提取文本特征...")
    
    # 存储所有文本特征 {item_id: text_feature}
    text_features_dict = {}
    processed_texts = 0
    error_texts = 0
    empty_texts = 0
    
    # 分批处理文本
    batch_size = 512
    item_ids = list(item_to_text.keys())
    text_contents = [item_to_text[item_id] for item_id in item_ids]
    
    for i in tqdm(range(0, len(item_ids), batch_size), desc="提取文本特征"):
        batch_item_ids = item_ids[i:i + batch_size]
        batch_texts = text_contents[i:i + batch_size]
        
        # 过滤空文本
        valid_indices = []
        valid_texts = []
        valid_item_ids = []
        
        for j, text in enumerate(batch_texts):
            if text and text.strip():  # 非空文本
                valid_indices.append(j)
                valid_texts.append(text)
                valid_item_ids.append(batch_item_ids[j])
            else:
                empty_texts += 1
        
        if not valid_texts:
            continue
            
        try:
            # 使用CLIP的tokenizer处理文本
            text_inputs = clip.tokenize(valid_texts, truncate=True).to(device)
            
            with torch.no_grad():
                batch_text_features = model.encode_text(text_inputs)
                batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            
            # 存储文本特征
            for j, item_id in enumerate(valid_item_ids):
                text_features_dict[item_id] = batch_text_features[j].cpu()
                processed_texts += 1
                
        except Exception as e:
            print(f"批量文本处理错误: {e}")
            # 回退到逐个处理
            for j, item_id in enumerate(valid_item_ids):
                try:
                    text = valid_texts[j]
                    text_input = clip.tokenize([text], truncate=True).to(device)
                    
                    with torch.no_grad():
                        text_feature = model.encode_text(text_input)
                        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                    
                    text_features_dict[item_id] = text_feature.cpu().squeeze(0)
                    processed_texts += 1
                except Exception as e2:
                    error_texts += 1
                    if error_texts <= 5:
                        print(f"文本处理错误: {item_id}, 错误: {str(e2)}")
    
    print(f"成功提取 {processed_texts} 个文本特征")
    print(f"空文本: {empty_texts}, 处理错误: {error_texts}")
    
    # 保存原始文本特征字典
    torch.save(text_features_dict, 'text_features_dict.pt')
    print(f"原始文本特征字典已保存到 text_features_dict.pt")
    
    # 第四步：根据item_map创建最终文本特征张量
    print("根据item_map整理文本特征...")
    text_features_final = torch.zeros(num_items, feature_dim)
    missing_text_items = []
    
    for item_id, idx in tqdm(item_map.items(), desc="整理文本特征"):
        if item_id in text_features_dict:
            text_features_final[idx] = text_features_dict[item_id]
        else:
            missing_text_items.append(item_id)
    
    # 保存最终文本特征张量
    torch.save(text_features_final, '/home/asaliao/APCL/dataset/IQON3000/feat/clip_text_features_indexedtensor')
    print(f"最终文本特征张量已保存到 clip_text_features_indexedtensor")
    print(f"文本特征张量形状: {text_features_final.shape}")
    print(f"缺失文本特征的产品数量: {len(missing_text_items)}")
    
    # 保存缺失产品列表
    with open('missing_text_items.json', 'w') as f:
        json.dump(missing_text_items, f, indent=2)
    
    return text_features_final, text_features_dict

def extract_text_from_json(json_data):
    """从JSON数据中提取文本内容"""
    text_parts = []
    
    # 根据实际JSON结构调整这些字段
    possible_fields = [
        'title', 'name', 'description', 'caption', 
        'text', 'content', 'detail', 'info',
        '商品名', '説明', '詳細', 'タイトル'  # 日语字段
    ]
    
    for field in possible_fields:
        if field in json_data and json_data[field]:
            text_parts.append(str(json_data[field]))
    
    # 如果找不到标准字段，尝试提取所有字符串值
    if not text_parts:
        for key, value in json_data.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
            elif isinstance(value, (list, dict)):
                # 递归处理嵌套结构
                nested_text = extract_text_from_nested(value)
                if nested_text:
                    text_parts.append(nested_text)
    
    # 合并所有文本部分
    combined_text = " ".join(text_parts)
    
    # 清理文本：移除多余空格和特殊字符
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    
    return combined_text if combined_text else ""

def extract_text_from_nested(data):
    """从嵌套数据结构中提取文本"""
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        texts = []
        for item in data:
            texts.append(extract_text_from_nested(item))
        return " ".join(filter(None, texts))
    elif isinstance(data, dict):
        texts = []
        for value in data.values():
            texts.append(extract_text_from_nested(value))
        return " ".join(filter(None, texts))
    else:
        return ""

def verify_text_features():
    """验证文本特征结果"""
    print("\n=== 文本特征验证 ===")
    
    try:
        text_features = torch.load('/home/asaliao/APCL/dataset/IQON3000/feat/clip_text_features_indexedtensor')
        print(f"最终文本特征形状: {text_features.shape}")
        
        # 验证特征质量
        norms = torch.norm(text_features, dim=1)
        zero_features = (norms == 0).sum().item()
        valid_features = text_features.shape[0] - zero_features
        
        print(f"有效文本特征数量: {valid_features}")
        print(f"零文本特征数量: {zero_features}")
        print(f"文本特征范数范围: [{norms.min().item():.4f}, {norms.max().item():.4f}]")
        
    except FileNotFoundError:
        print("文本特征文件不存在")
    
    # 检查缺失情况
    try:
        with open('missing_text_items.json', 'r') as f:
            missing_items = json.load(f)
        print(f"缺失文本特征的产品数量: {len(missing_items)}")
        if missing_items:
            print(f"前5个缺失产品: {missing_items[:5]}")
    except FileNotFoundError:
        print("缺失产品列表不存在")

if __name__ == "__main__":
    print("=== 文本特征提取 ===")
    print("注意：确保JSON文件包含产品的文本描述信息")
    
    # 提取文本特征
    text_features_final, text_features_dict = extract_text_features()
    
    # 验证结果
    verify_text_features()
    
    print("\n=== 使用说明 ===")
    print("1. text_features_dict.pt: 原始文本特征字典")
    print("2. clip_text_features_indexedtensor: 按item_map顺序排列的文本特征张量")
    print("3. missing_text_items.json: 缺失文本特征的产品列表")
    print("4. 使用方式: text_features = torch.load('/home/asaliao/APCL/dataset/IQON3000/feat/clip_text_features_indexedtensor')")