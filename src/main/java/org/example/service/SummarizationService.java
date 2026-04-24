package org.example.service;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class SummarizationService {

    @Autowired
    private ChatService chatService;

    private DashScopeChatModel chatModel;

    private static final String SUMMARIZE_TEMPLATE = """
    你是一个记忆管理专家。请将以下对话历史进行压缩总结，总结结果要不少于50字，不超过100字。
    保留关键的事实、决策和用户偏好，剔除寒暄和冗余信息。
    输出格式：[对话背景摘要]：{总结内容}
    ---
    对话历史：
    {history}
    """;

    @PostConstruct
    public void init() {
        chatModel = DashScopeChatModel.builder()
                .dashScopeApi(chatService.createDashScopeApi())
                .defaultOptions(DashScopeChatOptions.builder()
                        .withModel("qwen-max")
                        .withTemperature(0.7)
                        .withMaxToken(2000)
                        .withTopP(0.9)
                        .build()
                ).build();
    }

    public String summarize(String summhistory) {
        String finalPromptString = SUMMARIZE_TEMPLATE.replace("{history}", summhistory);
        Prompt prompt = new Prompt(finalPromptString);
        ChatResponse response = chatModel.call(prompt);
        String result = response.getResult().getOutput().getText();

        return result;
    }

}