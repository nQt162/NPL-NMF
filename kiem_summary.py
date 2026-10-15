import xml.etree.ElementTree as ET

def parse_vsolscsum(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    dataset = []
    # Duyệt qua từng bài viết trong file XML
    for doc in root.findall('.//doc') or root.findall('.//document'):
        full_sentences = []
        summary_sentences = []
        
        # Chỉ lấy các câu trong nội dung bài báo (bỏ qua phần comment)
        content_node = doc.find('content') or doc.find('sentences') or doc
        for sent in content_node.findall('.//s') or content_node.findall('.//sentence'):
            text = (sent.findtext('content') or sent.text or "").strip()
            if not text:
                continue
                
            # Lấy điểm vote (1-5) từ thuộc tính hoặc thẻ con
            label = sent.get('label') or sent.get('vote') or sent.findtext('label') or "0"
            score = int(label) if str(label).isdigit() else 0
            
            full_sentences.append(text)
            if score >= 3:  # Chuẩn của dataset: >= 3 đồng ý là câu tóm tắt
                summary_sentences.append(text)
                
        if full_sentences and summary_sentences:
            dataset.append({
                "full_text": " ".join(full_sentences),
                "extractive_summary": " ".join(summary_sentences),
                "num_orig_sents": len(full_sentences),
                "num_sum_sents": len(summary_sentences)
            })
            
    return dataset

# Chạy thử và in bài đầu tiên
data = parse_vsolscsum("F:\Downloads\VSoSLCSum.xml")
print(f"Đã trích xuất {len(data)} bài viết.")
print("\n--- BẢN GỐC BÀI 1 ---\n", data[0]["full_text"])
print("\n--- BẢN TÓM TẮT CHUẨN BÀI 1 ---\n", data[0]["extractive_summary"])