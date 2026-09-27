# TODO: Định nghĩa request/response Pydantic cho simulate, summarize, evaluate.
# TODO: Giới hạn text khoảng 20.000 ký tự, k trong 1..10,
#       summary_sentences trong 1..5, iterations trong 1..20.
# TODO: Mặc định k=2, summary_sentences=3, iterations=10, seed=42,
#       alpha=1, beta=1, gamma=0.5 theo TopicSum_AGENT.md.
# TODO: Mô tả snapshot {iteration,W,H,WH,loss}; sentence_analysis gồm index,
#       text, selected, relevance, coverage_gain, redundancy, score,
#       dominant_topic. Cho phép fallback_reason là null hoặc chuỗi.
