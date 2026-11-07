# 三相机图片转 JSON 观察结果

`qwen_images_to_json.py` 将保存的三张 JPEG 图片发送到阿里云百炼的 Qwen-VL 模型，识别前方视角中的桌面物品，并结合左右夹爪视角判断持物状态，生成结构化 JSON。

这是 **RGB 图片的离线分析报告**，不提供三维定位、同步采集时间或实时机器人控制信息，不能直接作为机器人执行依据。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `qwen_images_to_json.py` | 请求模型、校验返回结构并保存 JSON 的入口脚本 |
| `right_front_surround_camera.jpg` | 前方视角，角色 `front`，用于桌面物品识别 |
| `left_arm.jpg` | 左夹爪视角，角色 `left_hand`，用于判断持物 |
| `right_arm.jpg` | 右夹爪视角，角色 `right_hand`，用于判断持物 |
| `observation_all.json` | 更完整的多实例识别示例，包含名称、类别及描述 |

示例为已有模型输出，不是人工标注的真值。当前格式参考 `observation_all.json`，其中识别内容仍需人工核对。

## 环境与运行

需要 Python 3，仅使用标准库，无需安装第三方包。需要可用的百炼 API Key、模型访问权限和网络连接。运行时会将三张图片发送到所配置的百炼服务。

在仓库根目录使用 PowerShell 运行，交互输入 API Key（输入不回显）：

```powershell
python .\json_output\qwen_images_to_json.py --prompt-key --output .\json_output\observation_new.json
```

也可通过环境变量 `DASHSCOPE_API_KEY` 提供密钥。不要将真实密钥写入代码或提交到 Git。

脚本默认模型为 `qwen3-vl-plus`，默认地址是代码中配置的北京区域 MaaS 工作空间地址。该地址需要与你的账号、区域和权限相匹配，可显式指定地址和模型，例如：

```powershell
python .\json_output\qwen_images_to_json.py --prompt-key --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen3-vl-plus --output .\json_output\observation_new.json
```

输出文件必须不存在。建议每次指定新的 `--output` 文件名；若默认输出 `observation.json` 已存在，脚本会报错以避免覆盖。

## 参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `--input` | 脚本所在目录；目录中须包含上述三个固定文件名的 JPEG |
| `--output` | 脚本目录下的 `observation.json`；父目录会自动创建 |
| `--model` | 环境变量 `DASHSCOPE_MODEL`，未设置则为 `qwen3-vl-plus` |
| `--base-url` | 环境变量 `DASHSCOPE_BASE_URL`，未设置则使用脚本内置地址 |
| `--prompt-key` | 在交互终端中输入密钥；未指定时读取 `DASHSCOPE_API_KEY` |

地址必须使用 HTTPS，路径为 `/compatible-mode/v1`，主机限于脚本允许的官方百炼域名。脚本禁止 HTTP 重定向。每张图片最大 12 MiB，且会检查 JPEG 文件头；请求超时为 90 秒。

## 输出结构

```json
{
  "timestamp_ns": null,
  "detections": {
    "spoon_01": {
      "object_id": "spoon_01",
      "name": "勺子",
      "category": "spoon",
      "description": "银色长柄勺子，位于桌面前部",
      "position_m": null,
      "confidence": 0.9,
      "frame_id": null
    }
  },
  "feedback": {
    "held_object": "unknown",
    "spoon_loaded": null,
    "bowl_loaded": null
  }
}
```

- `detections`：最多 80 个物品实例，同类物品分别记录，例如 `bowl_01`、`bowl_02`。仅从前方图片识别桌面物品，左右夹爪图不补充物品列表。
- `object_id`：与实例字典键一致；`name` 为中文名称，`category` 为英文类别，`description` 描述外观和图片中的方位。
- `confidence`：模型自评的 0–1 数值，不是经过校准的成功概率。
- `timestamp_ns`、`position_m`、`frame_id`：当前固定为 `null`，因为保存的 RGB 图片不能提供可靠的同步时间、三维位置和坐标系。
- `held_object`：明确只有一只夹爪持物时记录物品名称，勺和碗使用 `spoon`、`bowl`；两只均明确空手为 `null`，任一夹爪状态未知为 `unknown`，两只均持物为 `multiple`。
- `spoon_loaded`、`bowl_loaded`：清晰看到有食物为 `true`，清晰看到为空为 `false`，遮挡、未出现或无法判断为 `null`。

脚本会校验 JSON 顶层字段、重复键、实例数、字符串长度、置信度和反馈类型。校验不能保证识别事实正确。成功时退出码为 `0`；请求、文件或结构校验失败时退出码为 `1`，错误写入标准错误输出。

## 常见问题

- `Output exists`：换一个尚不存在的 `--output` 文件名。
- 找不到图片：检查 `--input` 目录及三个固定文件名。
- HTTP 错误：检查 API Key、服务区域、工作空间地址、模型权限及额度。
- JSON 校验失败或响应未正常结束：脚本不会保存该次结果，核查配置后重新运行。

离线查看参数可运行 `python .\json_output\qwen_images_to_json.py --help`，不需要 API Key，也不会调用模型。
