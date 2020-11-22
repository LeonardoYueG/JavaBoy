**自定义gradle文件**：主model和多个副model，为了方便管理所有build.gradle的各个版本的管理

**优点：**

- 统一性
- 便于管理
- 版本管理



**Gradle构建的三个性能指标**

 * 全量编译：全部编译 - Open Project
 * 代码增量编译：修改了Java/Kotlin下面的代码的时候编译
 * 资源增量编译：修改了res下面的资源文件的时候编译