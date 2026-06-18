package org.example.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.MutationResult;
import io.milvus.param.R;
import io.milvus.param.RpcStatus;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.DeleteParam;
import io.milvus.param.dml.InsertParam;
import lombok.Getter;
import lombok.Setter;
import org.example.constant.MilvusConstants;
import org.example.dto.DocumentChunk;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;

/**
 * 向量索引服务
 * 负责读取文件、生成向量、存储到 Milvus
 */
@Service
public class VectorIndexService {

    private static final Logger logger = LoggerFactory.getLogger(VectorIndexService.class);

    @Autowired
    private MilvusServiceClient milvusClient;

    @Autowired
    private VectorEmbeddingService embeddingService;

    @Autowired
    private DocumentChunkService chunkService;

    @Value("${file.upload.path}")
    private String uploadPath;

    /**
     * 索引指定目录下的所有文件
     *
     * @param directoryPath 目录路径（可选，默认使用配置的上传目录）
     * @return 索引结果  这里可以优化：定时重建目录下所有文件的索引
     */
    public IndexingResult indexDirectory(String directoryPath) {
        IndexingResult result = new IndexingResult();
        result.setStartTime(LocalDateTime.now());

        try {
            // 使用指定目录或默认上传目录
            String targetPath = (directoryPath != null && !directoryPath.trim().isEmpty())
                    ? directoryPath : uploadPath;

            Path dirPath = Paths.get(targetPath).normalize();
            File directory = dirPath.toFile();

            if (!directory.exists() || !directory.isDirectory()) {
                throw new IllegalArgumentException("目录不存在或不是有效目录: " + targetPath);
            }

            result.setDirectoryPath(directory.getAbsolutePath());

            // 获取所有支持的文件
            File[] files = directory.listFiles((dir, name) ->
                    name.endsWith(".txt") || name.endsWith(".md")
            );

            if (files == null || files.length == 0) {
                logger.warn("目录中没有找到支持的文件: {}", targetPath);
                result.setTotalFiles(0);
                result.setSuccess(true);
                result.setEndTime(LocalDateTime.now());
                return result;
            }

            result.setTotalFiles(files.length);
            logger.info("开始索引目录: {}, 找到 {} 个文件", targetPath, files.length);

            // 遍历并索引每个文件
            for (File file : files) {
                try {
                    indexSingleFile(file.getAbsolutePath());
                    result.incrementSuccessCount();
                    logger.info("✓ 文件索引成功: {}", file.getName());
                } catch (Exception e) {
                    result.incrementFailCount();
                    result.addFailedFile(file.getAbsolutePath(), e.getMessage());
                    logger.error("✗ 文件索引失败: {}", file.getName(), e);
                }
            }

            result.setSuccess(result.getFailCount() == 0);
            result.setEndTime(LocalDateTime.now());

            logger.info("目录索引完成: 总数={}, 成功={}, 失败={}",
                    result.getTotalFiles(), result.getSuccessCount(), result.getFailCount());

            return result;

        } catch (Exception e) {
            logger.error("索引目录失败", e);
            result.setSuccess(false);
            result.setErrorMessage(e.getMessage());
            result.setEndTime(LocalDateTime.now());
            return result;
        }
    }

    /**
     * 索引单个文件
     *
     * @param filePath 文件路径
     * @throws Exception 索引失败时抛出异常
     */
    public void indexSingleFile(String filePath) throws Exception {
        Path path = Paths.get(filePath).normalize();
        File file = path.toFile();

        if (!file.exists() || !file.isFile()) {
            throw new IllegalArgumentException("文件不存在: " + filePath);
        }

        logger.info("开始索引文件: {}", path);

        // 1. 读取文件内容
        String content = Files.readString(path);
        logger.info("读取文件: {}, 内容长度: {} 字符", path, content.length());

        // 2. 删除该文件的旧数据（如果存在）
        deleteExistingData(path.toString());

        // 3. 文档分片
        List<DocumentChunk> chunks = chunkService.chunkDocument(content, path.toString());
        logger.info("文档分片完成: {} -> {} 个分片", filePath, chunks.size());

        // 4. 为每个分片生成向量并插入 Milvus
        for (int i = 0; i < chunks.size(); i++) {
            DocumentChunk chunk = chunks.get(i);

            try {
                // 生成向量
                List<Float> vector = embeddingService.generateEmbedding(chunk.getContent());
                validateVector(vector, filePath, chunk.getChunkIndex());
//TODO: metaData的作用

/*                metadata.put("source", path.toString()); // 来自哪个文件
                metadata.put("chunkIndex", chunk.getIndex()); // 第几块
                metadata.put("totalChunks", chunks.size()); // 总共几块
                metadata.put("content", chunk.getText()); // 原始文本（可选）
                这几个字段名我设置milvus数据库的schema的时候要加上吗

                是的，必须加。
                Milvus 是强 Schema 的，字段不提前声明就无法写入。
                CollectionSchemaParam schema = CollectionSchemaParam.newBuilder()
                        .withFieldTypes(Arrays.asList(
                                // 必须有的系统字段
                                FieldType.newBuilder()
                                        .withName("id")
                                        .withDataType(DataType.Int64)
                                        .withPrimaryKey(true)
                                        .withAutoID(true)
                                        .build(),
                                // 向量字段
                                FieldType.newBuilder()
                                        .withName("vector")
                                        .withDataType(DataType.FloatVector)
                                        .withDimension(1536) // 你的 embedding 维度
                                        .build(),
                                // 元数据字段 ↓
                                FieldType.newBuilder()
                                        .withName("source")
                                        .withDataType(DataType.VarChar)
                                        .withMaxLength(1024)
                                        .build(),
                                FieldType.newBuilder()
                                        .withName("chunkIndex")
                                        .withDataType(DataType.Int64)
                                        .build(),
                                FieldType.newBuilder()
                                        .withName("totalChunks")
                                        .withDataType(DataType.Int64)
                                        .build(),
                                FieldType.newBuilder()
                                        .withName("content")
                                        .withDataType(DataType.VarChar)
                                        .withMaxLength(65535)
                                        .build()
                        ))
                        .build();


                最重要的还是content也就是文章分块之后的内容
                其他的都是这一行数据所对应实体的属性罢了
                比如来源也就是路径
                还有这一段content对应的id，分块之后向量化之后的内容存到vector 。。
                剩下的chunkindex和totalchunks是干什么的 ，
                我猜测chunkindex的作用应该是当时做语义传递的时候找下一个chunk。
                那totalchunk干什么 每一个实体这个属性不都是一样的吗

                21:00
                你猜对了一半，直接说结论：
                chunkIndex 的真实用途
                不只是找下一块，主要是重组和排序。
                检索时可能同时命中同一文档的第 3 块和第 7 块，没有 chunkIndex 你不知道谁在前谁在后，拼回去顺序就乱了。
                totalChunks 的用途
                它的唯一价值：
                不用额外查询就能知道"这个文档总共被切成几块，我现在拿到的是第几块/共几块"
                典型场景是日志和调试，或者展示给用户看："已找到来源文档（第2块，共5块）"。
                生产环境很多人直接不存这个字段，完全可以删掉。*/

                // 构建元数据（包含文件信息） 这里只传递了三个变量, 分别是文件的路径,分块的编号, 块的总大小
                Map<String, Object> metadata = buildMetadata(path.toString(), chunk, chunks.size());

                // 插入到 Milvus
                insertToMilvus(chunk.getContent(), vector, metadata, chunk.getChunkIndex());

                logger.info("✓ 分片 {}/{} 索引成功", i + 1, chunks.size());

            } catch (Exception e) {
                logger.error("✗ 分片 {}/{} 索引失败", i + 1, chunks.size(), e);
                throw new RuntimeException("分片索引失败: " + e.getMessage(), e);
            }
        }

        logger.info("文件索引完成: {}, 共 {} 个分片", filePath, chunks.size());
    }

    /**
     * 删除文件的旧数据（根据 metadata._source）
     */
    private void deleteExistingData(String filePath) {
        try {
            // 使用统一的路径分隔符（正斜杠）用于Milvus存储，避免表达式解析错误
            // 将系统路径转换为统一格式
            Path path = Paths.get(filePath).normalize();
            String normalizedPath = path.toString().replace(File.separator, "/");

            // 构建删除表达式：metadata["_source"] == "xxx"
            String expr = String.format("metadata[\"_source\"] == \"%s\"", normalizedPath);

            logger.info("准备删除旧数据，路径: {}, 表达式: {}", normalizedPath, expr);

            // 确保 collection 已加载（删除操作需要集合已加载）
            R<RpcStatus> loadResponse = milvusClient.loadCollection(
                    LoadCollectionParam.newBuilder()
                            .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
                            .build()
            );

            // 状态码 65535 表示集合已经加载，这不是错误
            if (loadResponse.getStatus() != 0 && loadResponse.getStatus() != 65535) {
                logger.warn("加载 collection 失败: {}", loadResponse.getMessage());
                return;
            }

            DeleteParam deleteParam = DeleteParam.newBuilder()
                    .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
                    .withExpr(expr)
                    .build();

            R<MutationResult> response = milvusClient.delete(deleteParam);

            if (response.getStatus() != 0) {
                logger.warn("删除旧数据时出现警告: {}", response.getMessage());
            } else {
                long deletedCount = response.getData().getDeleteCnt();
                logger.info("✓ 已删除文件的旧数据: {}, 删除记录数: {}", normalizedPath, deletedCount);
            }

        } catch (Exception e) {
            logger.warn("删除旧数据失败（可能是首次索引）: {}", e.getMessage());
        }
    }

    /**
     * 构建元数据（包含文件信息）
     */
    private Map<String, Object> buildMetadata(String filePath, DocumentChunk chunk, int totalChunks) {
        Map<String, Object> metadata = new HashMap<>();

        // 标准化路径：使用统一的路径分隔符（正斜杠）用于存储，确保跨平台一致性
        Path path = Paths.get(filePath).normalize();
        String normalizedPath = path.toString().replace(File.separator, "/");

        // 文件信息
        Path fileName = path.getFileName();
        String fileNameStr = fileName != null ? fileName.toString() : "";
        String extension = "";
        int dotIndex = fileNameStr.lastIndexOf('.');
        if (dotIndex > 0) {
            extension = fileNameStr.substring(dotIndex);
        }

        metadata.put("_source", normalizedPath);
        metadata.put("_extension", extension);
        metadata.put("_file_name", fileNameStr);

        // 分片信息
        metadata.put("chunkIndex", chunk.getChunkIndex());
        metadata.put("totalChunks", totalChunks);

        // 标题信息
        if (chunk.getTitle() != null && !chunk.getTitle().isEmpty()) {
            metadata.put("title", chunk.getTitle());
        }

        return metadata;
    }

    /**
     * 插入向量到 Milvus
     */
    private void insertToMilvus(String content, List<Float> vector,
                                Map<String, Object> metadata, int chunkIndex) throws Exception {
        try {
            // 确保 collection 已加载
            R<RpcStatus> loadResponse = milvusClient.loadCollection(
                    LoadCollectionParam.newBuilder()
                            .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
                            .build()
            );

            if (loadResponse.getStatus() != 0 && loadResponse.getStatus() != 65535) {
                throw new RuntimeException("加载 collection 失败: " + loadResponse.getMessage());
            }

            // 生成唯一 ID（使用 _source + 分片索引）
            String source = (String) metadata.get("_source");
            String id = UUID.nameUUIDFromBytes((source + "_" + chunkIndex).getBytes()).toString();

            // 构建字段数据
            List<InsertParam.Field> fields = new ArrayList<>();

            //相当于schema
            // ID 字段
            fields.add(new InsertParam.Field("id", Collections.singletonList(id)));

            // content 字段
            fields.add(new InsertParam.Field("content", Collections.singletonList(content)));

            // vector 字段
            fields.add(new InsertParam.Field(MilvusConstants.VECTOR_FIELD_NAME, Collections.singletonList(vector)));

            // metadata 字段（JSON 对象）
            com.google.gson.Gson gson = new com.google.gson.Gson();
            com.google.gson.JsonObject metadataJson = gson.toJsonTree(metadata).getAsJsonObject();
            fields.add(new InsertParam.Field("metadata", Collections.singletonList(metadataJson)));

            // 构建插入参数
            InsertParam insertParam = InsertParam.newBuilder()
                    .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
                    .withFields(fields)
                    .build();

            // 执行插入  这里用的是按列插入InsertParam requestParam   这个是按行插入InsertRowsParam requestParam
            R<MutationResult> insertResponse = milvusClient.insert(insertParam);

            if (insertResponse.getStatus() != 0) {
                throw new RuntimeException("插入向量失败: " + insertResponse.getMessage());
            }

            logger.debug("向量插入成功: id={}, source={}, chunk={}", id, source, chunkIndex);

        } catch (Exception e) {
            logger.error("插入向量到 Milvus 失败", e);
            throw e;
        }
    }

    /**
     * 校验向量有效性，避免向量空值导致 Milvus 报 got nil。
     */
    private void validateVector(List<Float> vector, String filePath, int chunkIndex) {
        if (vector == null || vector.isEmpty()) {
            throw new IllegalStateException(String.format(
                    "向量为空，无法写入 Milvus。file=%s, chunk=%d", filePath, chunkIndex));
        }
        if (vector.size() != MilvusConstants.VECTOR_DIM) {
            throw new IllegalStateException(String.format(
                    "向量维度不匹配，期望=%d，实际=%d。file=%s, chunk=%d",
                    MilvusConstants.VECTOR_DIM, vector.size(), filePath, chunkIndex));
        }
    }

    /**
     * 索引结果类
     */
    @Getter
    public static class IndexingResult {
        @Setter
        private boolean success;
        @Setter
        private String directoryPath;
        @Setter
        private int totalFiles;
        private int successCount;
        private int failCount;
        @Setter
        private LocalDateTime startTime;
        @Setter
        private LocalDateTime endTime;
        @Setter
        private String errorMessage;
        private Map<String, String> failedFiles = new HashMap<>();

        public void incrementSuccessCount() {
            this.successCount++;
        }

        public void incrementFailCount() {
            this.failCount++;
        }

        public long getDurationMs() {
            if (startTime != null && endTime != null) {
                return java.time.Duration.between(startTime, endTime).toMillis();
            }
            return 0;
        }

        public void addFailedFile(String filePath, String error) {
            this.failedFiles.put(filePath, error);
        }
    }
}
