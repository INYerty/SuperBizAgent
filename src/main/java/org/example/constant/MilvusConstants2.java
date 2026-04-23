package org.example.constant;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;

public class MilvusConstants2 {

    /**
     * Milvus 数据库名称
     */
    public static final String MILVUS_DB_NAME = "inyert_DB";

    /**
     * Milvus 集合名称
     */
    public static final String MILVUS_COLLECTION_NAME = "test_01";

    /**
     * 向量维度（豆包 embedding 模型的维度）
     */
    public static final int VECTOR_DIM = 1024;  // 豆包模型返回1024维向量

    /**
     * ID字段最大长度
     */
    public static final int ID_MAX_LENGTH = 100;

    /**
     * Content字段最大长度
     */
    public static final int CONTENT_MAX_LENGTH = 50000;

    /**
     * 默认分片数
     */
    public static final int DEFAULT_SHARD_NUMBER = 1;


    private MilvusConstants2() {
        // 工具类，禁止实例化
    }
}
