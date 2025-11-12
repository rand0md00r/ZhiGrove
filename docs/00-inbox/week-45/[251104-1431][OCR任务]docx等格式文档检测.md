# 数据分布排查
1. 数据类型

> 只处理doc，docx，pdf，txt文档；

| 子目录 | 文件类型 |
| - | - |
| (Test Bank)Building Java Programs A Back to Basics Approach 5th Edition | doc 54, html 18, pdf 30, txt 13 |
| (Test Bank)Calculus Multivariable 7th Edition by Hughes-Hallett | pdf 10 |
| (Test Bank)Chemistry & Chemical Reactivity , 10th Edition John C. Kotz | docx 26 |
| (Test Bank)Data Analysis and Decision Making 4th Edition by Albright | rtf 16 |
| (Test Bank)Data and Computer Communications,10th International Edition | doc 50 |
| (Test Bank)ECON MICRO, 5th Edition | jpeg 179, png 127, xml 42, zip 21 |
| (Test Bank)Introduction to Java Programming and Data Structures, Comprehensive Version, 11th Edition | txt 44 |
| (Test Bank)Math for Business and Finance An Algebraic Approach 2nd Edition 2e | docx 42 |
| (Test Bank)Mathematical Ideas 13th Edition by Miller | pdf 15 |
| Big Java Early Objects 6E - Horstmann - TB | css 1, xhtml 26 |

2. 解析策略
    1. 基础解析层为每种格式封装统一接口，工具库调用：
        - docx：python-docx / docx2txt；
        - doc：libreoffice / antiword解析，或 mammoth / pypandoc 进行格式转换；
        - pdf：pdfminer.six / PyMuPDF / pdfplumber；
    
        所有解析函数返回统一数据结构；

    2. 策略层
        根据数据来源定义策略描述，用配置文件或数据库保存。

    3. 流水线框架
        调度器，根据来源选择策略，依次执行“读取 -> 预处理 -> 文本解析 -> 内容结构化 -> 导出“。组件化开发：PreProcessor，Parser，Postprocessor；

    4. 