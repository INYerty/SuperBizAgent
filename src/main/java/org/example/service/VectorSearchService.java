package org.example.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.Getter;
import lombok.Setter;
import org.example.constant.MilvusConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 向量搜索服务
 * 负责从 Milvus 中搜索相似向量
 */
@Service
public class VectorSearchService {

    private static final Logger logger = LoggerFactory.getLogger(VectorSearchService.class);

    @Autowired
    private MilvusServiceClient milvusClient;

    @Autowired
    private VectorEmbeddingService embeddingService;

    /**
     * 搜索相似文档
     * 
     * @param query 查询文本
     * @param topK 返回最相似的K个结果
     * @return 搜索结果列表
     */
    public List<SearchResult> searchSimilarDocuments(String query, int topK) {
        try {
            logger.info("开始搜索相似文档, 查询: {}, topK: {}", query, topK);

            // 1. 将查询文本向量化
            List<Float> queryVector = embeddingService.generateQueryVector(query);
            if (queryVector == null || queryVector.isEmpty()) {
                throw new IllegalStateException("查询向量为空，无法执行检索");
            }
            if (queryVector.size() != MilvusConstants.VECTOR_DIM) {
                throw new IllegalStateException("查询向量维度不匹配，期望="
                        + MilvusConstants.VECTOR_DIM + "，实际=" + queryVector.size());
            }
            logger.debug("查询向量生成成功, 维度: {}", queryVector.size());

            // 2. 构建搜索参数
            SearchParam searchParam = SearchParam.newBuilder()
                    .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
                    .withVectorFieldName(MilvusConstants.VECTOR_FIELD_NAME)
                    // 这里可以传 多个查询向量 多个查询的向量是根据问题让LLM进行分析得来的
                    // 假如我跟LLM说我要将问题划分三个维度进行匹配 将问题分为三个content 存入List 调用批量生成向量的方法
                    // 那么 我就会得到三个向量 类型是原来是List<Float[]> vectors转换后是List<List<Float>> vectors
                    // 最后将他传入到withVectors(vectors)这个方法就能进行搜索了。
                    .withVectors(Collections.singletonList(queryVector))
                    .withTopK(topK)
                    .withMetricType(io.milvus.param.MetricType.L2)// 在建立索引的时候制定了搜索方法是L2 空间中 两条直线的距离 越小代表相关性越大
                    // 注意这里是输出的字段 with_outField 不是排除字段without_field;
                    .withOutFields(List.of("id", "content", "metadata"))
                    // 建索引的时候一共分为了128个桶，在检索时，先定位到离你的 query 向量最近的 $N$ 个桶，然后只在这 $N$ 个桶里进行全量搜索。
                    .withParams("{\"nprobe\":10}")
                    .build();

            // 3. 执行搜索
            R<SearchResults> searchResponse = milvusClient.search(searchParam);

            if (searchResponse.getStatus() != 0) {
                throw new RuntimeException("向量搜索失败: " + searchResponse.getMessage());
            }

            // 4. 解析搜索结果
            SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResponse.getData().getResults()); // 结果包装器
            List<SearchResult> results = new ArrayList<>(); // 存处理完成的搜索结果
            //这里指的是查询第0个向量的所有行,这里我就查了一个什么呢 这个！MilvusConstants.VECTOR_FIELD_NAME 也就遍历查到的所有行的数据
            for (int i = 0; i < wrapper.getRowRecords(0).size(); i++) {  // 遍历数据条数  wrapper.getRowRecords(0).size()其实就是topK
                SearchResult result = new SearchResult();
                //先拿到这个向量查到的
                result.setId((String) wrapper.getIDScore(0).get(i).get("id"));
                result.setContent((String) wrapper.getFieldData("content", 0).get(i));
                result.setScore(wrapper.getIDScore(0).get(i).getScore());
                
                // 解析 metadata
                Object metadataObj = wrapper.getFieldData("metadata", 0).get(i);
                if (metadataObj != null) {
                    result.setMetadata(metadataObj.toString());
                }
                
                results.add(result);
            }
//------------------------TODO-----4/18------------
            logger.info("搜索完成, 找到 {} 个相似文档", results.size());
            return results;

        } catch (Exception e) {
            logger.error("搜索相似文档失败", e);
            throw new RuntimeException("搜索失败: " + e.getMessage(), e);
        }
    }

    /**
     * 搜索结果类
     */
    @Setter
    @Getter
    public static class SearchResult {
        private String id;
        private String content;
        private float score;
        private String metadata;

    }
}
