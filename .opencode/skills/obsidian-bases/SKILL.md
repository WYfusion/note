---
name: obsidian-bases
description: 优化的Obsidian Bases数据库系统，专为学习笔记管理设计。提供强大的笔记查询、分类、统计和可视化功能，帮助构建高效的知识管理体系。
---

# Obsidian学习笔记数据库技能

## 🎯 设计理念

为学习笔记优化的数据库系统，实现：
- **智能分类**: 按学科、难度、状态自动分组
- **进度追踪**: 实时监控学习进度和掌握情况
- **知识关联**: 发现笔记间的关联关系
- **统计分析**: 可视化学习数据和趋势

## 📊 核心功能

Obsidian Bases = YAML配置 + 动态查询 + 多视图展示
- **数据源**: 笔记属性和元数据
- **查询引擎**: 强大的过滤和公式系统
- **视图系统**: 表格、卡片、列表等多种展示
- **统计分析**: 自动计算和汇总功能

## File Format

Base files use the `.base` extension and contain valid YAML. They can also be embedded in Markdown code blocks.

## Complete Schema

```yaml
# Global filters apply to ALL views in the base
filters:
  # Can be a single filter string
  # OR a recursive filter object with and/or/not
  and: []
  or: []
  not: []

# Define formula properties that can be used across all views
formulas:
  formula_name: 'expression'

# Configure display names and settings for properties
properties:
  property_name:
    displayName: "Display Name"
  formula.formula_name:
    displayName: "Formula Display Name"
  file.ext:
    displayName: "Extension"

# Define custom summary formulas
summaries:
  custom_summary_name: 'values.mean().round(3)'

# Define one or more views
views:
  - type: table | cards | list | map
    name: "View Name"
    limit: 10                    # Optional: limit results
    groupBy:                     # Optional: group results
      property: property_name
      direction: ASC | DESC
    filters:                     # View-specific filters
      and: []
    order:                       # Properties to display in order
      - file.name
      - property_name
      - formula.formula_name
    summaries:                   # Map properties to summary formulas
      property_name: Average
```

## Filter Syntax

Filters narrow down results. They can be applied globally or per-view.

### Filter Structure

```yaml
# Single filter
filters: 'status == "done"'

# AND - all conditions must be true
filters:
  and:
    - 'status == "done"'
    - 'priority > 3'

# OR - any condition can be true
filters:
  or:
    - 'file.hasTag("book")'
    - 'file.hasTag("article")'

# NOT - exclude matching items
filters:
  not:
    - 'file.hasTag("archived")'

# Nested filters
filters:
  or:
    - file.hasTag("tag")
    - and:
        - file.hasTag("book")
        - file.hasLink("Textbook")
    - not:
        - file.hasTag("book")
        - file.inFolder("Required Reading")
```

### Filter Operators

| Operator | Description |
|----------|-------------|
| `==` | equals |
| `!=` | not equal |
| `>` | greater than |
| `<` | less than |
| `>=` | greater than or equal |
| `<=` | less than or equal |
| `&&` | logical and |
| `\|\|` | logical or |
| <code>!</code> | logical not |

## Properties

### Three Types of Properties

1. **Note properties** - From frontmatter: `note.author` or just `author`
2. **File properties** - File metadata: `file.name`, `file.mtime`, etc.
3. **Formula properties** - Computed values: `formula.my_formula`

### File Properties Reference

| Property | Type | Description |
|----------|------|-------------|
| `file.name` | String | File name |
| `file.basename` | String | File name without extension |
| `file.path` | String | Full path to file |
| `file.folder` | String | Parent folder path |
| `file.ext` | String | File extension |
| `file.size` | Number | File size in bytes |
| `file.ctime` | Date | Created time |
| `file.mtime` | Date | Modified time |
| `file.tags` | List | All tags in file |
| `file.links` | List | Internal links in file |
| `file.backlinks` | List | Files linking to this file |
| `file.embeds` | List | Embeds in the note |
| `file.properties` | Object | All frontmatter properties |

### The `this` Keyword

- In main content area: refers to the base file itself
- When embedded: refers to the embedding file
- In sidebar: refers to the active file in main content

## Formula Syntax

Formulas compute values from properties. Defined in the `formulas` section.

```yaml
formulas:
  # Simple arithmetic
  total: "price * quantity"
  
  # Conditional logic
  status_icon: 'if(done, "✅", "⏳")'
  
  # String formatting
  formatted_price: 'if(price, price.toFixed(2) + " dollars")'
  
  # Date formatting
  created: 'file.ctime.format("YYYY-MM-DD")'
  
  # Complex expressions
  days_old: '((now() - file.ctime) / 86400000).round(0)'
```

## Functions Reference

### Global Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `date()` | `date(string): date` | Parse string to date. Format: `YYYY-MM-DD HH:mm:ss` |
| `duration()` | `duration(string): duration` | Parse duration string |
| `now()` | `now(): date` | Current date and time |
| `today()` | `today(): date` | Current date (time = 00:00:00) |
| `if()` | `if(condition, trueResult, falseResult?)` | Conditional |
| `min()` | `min(n1, n2, ...): number` | Smallest number |
| `max()` | `max(n1, n2, ...): number` | Largest number |
| `number()` | `number(any): number` | Convert to number |
| `link()` | `link(path, display?): Link` | Create a link |
| `list()` | `list(element): List` | Wrap in list if not already |
| `file()` | `file(path): file` | Get file object |
| `image()` | `image(path): image` | Create image for rendering |
| `icon()` | `icon(name): icon` | Lucide icon by name |
| `html()` | `html(string): html` | Render as HTML |
| `escapeHTML()` | `escapeHTML(string): string` | Escape HTML characters |

### Any Type Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `isTruthy()` | `any.isTruthy(): boolean` | Coerce to boolean |
| `isType()` | `any.isType(type): boolean` | Check type |
| `toString()` | `any.toString(): string` | Convert to string |

### Date Functions & Fields

**Fields:** `date.year`, `date.month`, `date.day`, `date.hour`, `date.minute`, `date.second`, `date.millisecond`

| Function | Signature | Description |
|----------|-----------|-------------|
| `date()` | `date.date(): date` | Remove time portion |
| `format()` | `date.format(string): string` | Format with Moment.js pattern |
| `time()` | `date.time(): string` | Get time as string |
| `relative()` | `date.relative(): string` | Human-readable relative time |
| `isEmpty()` | `date.isEmpty(): boolean` | Always false for dates |

### Date Arithmetic

```yaml
# Duration units: y/year/years, M/month/months, d/day/days, 
#                 w/week/weeks, h/hour/hours, m/minute/minutes, s/second/seconds

# Add/subtract durations
"date + \"1M\""           # Add 1 month
"date - \"2h\""           # Subtract 2 hours
"now() + \"1 day\""       # Tomorrow
"today() + \"7d\""        # A week from today

# Subtract dates for millisecond difference
"now() - file.ctime"

# Complex duration arithmetic
"now() + (duration('1d') * 2)"
```

### String Functions

**Field:** `string.length`

| Function | Signature | Description |
|----------|-----------|-------------|
| `contains()` | `string.contains(value): boolean` | Check substring |
| `containsAll()` | `string.containsAll(...values): boolean` | All substrings present |
| `containsAny()` | `string.containsAny(...values): boolean` | Any substring present |
| `startsWith()` | `string.startsWith(query): boolean` | Starts with query |
| `endsWith()` | `string.endsWith(query): boolean` | Ends with query |
| `isEmpty()` | `string.isEmpty(): boolean` | Empty or not present |
| `lower()` | `string.lower(): string` | To lowercase |
| `title()` | `string.title(): string` | To Title Case |
| `trim()` | `string.trim(): string` | Remove whitespace |
| `replace()` | `string.replace(pattern, replacement): string` | Replace pattern |
| `repeat()` | `string.repeat(count): string` | Repeat string |
| `reverse()` | `string.reverse(): string` | Reverse string |
| `slice()` | `string.slice(start, end?): string` | Substring |
| `split()` | `string.split(separator, n?): list` | Split to list |

### Number Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `abs()` | `number.abs(): number` | Absolute value |
| `ceil()` | `number.ceil(): number` | Round up |
| `floor()` | `number.floor(): number` | Round down |
| `round()` | `number.round(digits?): number` | Round to digits |
| `toFixed()` | `number.toFixed(precision): string` | Fixed-point notation |
| `isEmpty()` | `number.isEmpty(): boolean` | Not present |

### List Functions

**Field:** `list.length`

| Function | Signature | Description |
|----------|-----------|-------------|
| `contains()` | `list.contains(value): boolean` | Element exists |
| `containsAll()` | `list.containsAll(...values): boolean` | All elements exist |
| `containsAny()` | `list.containsAny(...values): boolean` | Any element exists |
| `filter()` | `list.filter(expression): list` | Filter by condition (uses `value`, `index`) |
| `map()` | `list.map(expression): list` | Transform elements (uses `value`, `index`) |
| `reduce()` | `list.reduce(expression, initial): any` | Reduce to single value (uses `value`, `index`, `acc`) |
| `flat()` | `list.flat(): list` | Flatten nested lists |
| `join()` | `list.join(separator): string` | Join to string |
| `reverse()` | `list.reverse(): list` | Reverse order |
| `slice()` | `list.slice(start, end?): list` | Sublist |
| `sort()` | `list.sort(): list` | Sort ascending |
| `unique()` | `list.unique(): list` | Remove duplicates |
| `isEmpty()` | `list.isEmpty(): boolean` | No elements |

### File Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `asLink()` | `file.asLink(display?): Link` | Convert to link |
| `hasLink()` | `file.hasLink(otherFile): boolean` | Has link to file |
| `hasTag()` | `file.hasTag(...tags): boolean` | Has any of the tags |
| `hasProperty()` | `file.hasProperty(name): boolean` | Has property |
| `inFolder()` | `file.inFolder(folder): boolean` | In folder or subfolder |

### Link Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `asFile()` | `link.asFile(): file` | Get file object |
| `linksTo()` | `link.linksTo(file): boolean` | Links to file |

### Object Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `isEmpty()` | `object.isEmpty(): boolean` | No properties |
| `keys()` | `object.keys(): list` | List of keys |
| `values()` | `object.values(): list` | List of values |

### Regular Expression Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `matches()` | `regexp.matches(string): boolean` | Test if matches |

## View Types

### Table View

```yaml
views:
  - type: table
    name: "My Table"
    order:
      - file.name
      - status
      - due_date
    summaries:
      price: Sum
      count: Average
```

### Cards View

```yaml
views:
  - type: cards
    name: "Gallery"
    order:
      - file.name
      - cover_image
      - description
```

### List View

```yaml
views:
  - type: list
    name: "Simple List"
    order:
      - file.name
      - status
```

### Map View

Requires latitude/longitude properties and the Maps community plugin.

```yaml
views:
  - type: map
    name: "Locations"
    # Map-specific settings for lat/lng properties
```

## Default Summary Formulas

| Name | Input Type | Description |
|------|------------|-------------|
| `Average` | Number | Mathematical mean |
| `Min` | Number | Smallest number |
| `Max` | Number | Largest number |
| `Sum` | Number | Sum of all numbers |
| `Range` | Number | Max - Min |
| `Median` | Number | Mathematical median |
| `Stddev` | Number | Standard deviation |
| `Earliest` | Date | Earliest date |
| `Latest` | Date | Latest date |
| `Range` | Date | Latest - Earliest |
| `Checked` | Boolean | Count of true values |
| `Unchecked` | Boolean | Count of false values |
| `Empty` | Any | Count of empty values |
| `Filled` | Any | Count of non-empty values |
| `Unique` | Any | Count of unique values |

## 🎓 学习笔记数据库示例

### 学习进度总览
```yaml
filters:
  and:
    - file.hasTag("学习")
    - 'file.ext == "md"'

formulas:
  # 学习进度计算
  progress_percent: 'if(progress, progress, 0)'
  mastery_level: 'if(confidence, "⭐".repeat(confidence), "📝")'
  urgency_score: 'if(priority == "高", 3, if(priority == "中", 2, 1))'
  days_since_created: '((now() - date(created)) / 86400000).round(0)'
  needs_review: 'if(next_review && date(next_review) < today(), "🔄", "")'

  # 智能分类
  category_type: 'if(file.hasTag("概念"), "📖 概念", if(file.hasTag("算法"), "⚙️ 算法", if(file.hasTag("实现"), "💻 实现", "📚 其他")))'

  # 难度可视化
  difficulty_emoji: 'if(difficulty == "入门", "🟢", if(difficulty == "进阶", "🟡", if(difficulty == "高级", "🔴", "⚪")))'

properties:
  status:
    displayName: "状态"
  difficulty:
    displayName: "难度"
  rating:
    displayName: "评分"
  formula.progress_percent:
    displayName: "进度%"
  formula.mastery_level:
    displayName: "掌握度"
  formula.category_type:
    displayName: "分类"
  formula.difficulty_emoji:
    displayName: ""
  formula.needs_review:
    displayName: ""

views:
  - type: table
    name: "📊 学习总览"
    order:
      - formula.category_type
      - file.name
      - status
      - formula.difficulty_emoji
      - formula.progress_percent
      - formula.mastery_level
      - rating
      - formula.needs_review
    groupBy:
      property: status
      direction: ASC
    summaries:
      formula.progress_percent: Average
      rating: Average

  - type: cards
    name: "🎯 当前学习"
    filters:
      and:
        - 'status == "学习中"'
    order:
      - file.name
      - formula.category_type
      - formula.difficulty_emoji
      - formula.progress_percent
      - formula.mastery_level
```

### 📚 知识体系管理
```yaml
filters:
  and:
    - or:
        - file.hasTag("概念")
        - file.hasTag("算法")
        - file.hasTag("理论")

formulas:
  # 知识关联度
  link_count: 'file.links.length'
  backlink_count: 'file.backlinks.length'
  knowledge_score: '(rating * confidence + link_count + backlink_count) / 3'

  # 学习路径
  learning_stage: 'if(progress < 30, "🌱 初学", if(progress < 70, "🌿 进阶", "🌳 精通"))'
  completion_rate: 'if(progress, progress, 0)'

  # 依赖关系
  has_prerequisites: 'if(prerequisites, "✅", "❌")'
  dependency_count: 'if(dependencies, dependencies.length, 0)'

  # 时间分析
  learning_duration: '((date(modified) - date(created)) / 86400000).round(0)'
  is_recent: 'if(date(modified) > now() - "7d", "🆕", "")'

properties:
  formula.learning_stage:
    displayName: "学习阶段"
  formula.knowledge_score:
    displayName: "知识分"
  formula.link_count:
    displayName: "链接数"
  formula.completion_rate:
    displayName: "完成度"
  formula.has_prerequisites:
    displayName: "前置"
  formula.learning_duration:
    displayName: "学习天数"
  formula.is_recent:
    displayName: ""

views:
  - type: table
    name: "🧠 知识图谱"
    order:
      - file.name
      - formula.learning_stage
      - formula.knowledge_score
      - formula.link_count
      - formula.backlink_count
      - formula.completion_rate
      - formula.has_prerequisites
      - formula.learning_duration
      - formula.is_recent
    groupBy:
      property: difficulty
      direction: ASC
    summaries:
      formula.knowledge_score: Average
      formula.completion_rate: Average

  - type: list
    name: "🎯 核心概念"
    filters:
      and:
        - file.hasTag("概念")
        - 'rating >= 4'
    order:
      - file.name
      - formula.knowledge_score
      - confidence
```

### 📈 学习统计分析
```yaml
filters:
  and:
    - file.hasTag("学习")
    - 'progress != ""'

formulas:
  # 统计指标
  total_notes: '1'
  completed_count: 'if(status == "已掌握", 1, 0)'
  in_progress_count: 'if(status == "学习中", 1, 0)'

  # 质量评估
  quality_score: '(rating + confidence + progress) / 3'
  is_high_quality: 'if(quality_score >= 4, "🏆", "")'

  # 效率分析
  efficiency: 'if(learning_duration > 0, (progress / learning_duration).round(2), 0)'
  is_efficient: 'if(efficiency >= 5, "⚡", "")'

  # 复习管理
  review_overdue: 'if(next_review && date(next_review) < today(), "⏰", "")'
  days_to_review: 'if(next_review, ((date(next_review) - today()) / 86400000).round(0), "")'

summaries:
  total_learning: 'values.sum()'
  completion_rate: '(values.filter(v => v == 1).sum() / values.sum() * 100).round(1)'
  avg_quality: 'values.mean().round(2)'
  avg_efficiency: 'values.mean().round(2)'

properties:
  formula.quality_score:
    displayName: "质量分"
  formula.efficiency:
    displayName: "效率"
  formula.is_high_quality:
    displayName: ""
  formula.is_efficient:
    displayName: ""
  formula.review_overdue:
    displayName: "复习"
  formula.days_to_review:
    displayName: "天数"

views:
  - type: table
    name: "📊 学习统计"
    order:
      - file.name
      - status
      - formula.quality_score
      - formula.efficiency
      - progress
      - formula.review_overdue
      - formula.days_to_review
      - formula.is_high_quality
      - formula.is_efficient
    summaries:
      formula.total_notes: total_learning
      formula.completed_count: completion_rate
      formula.quality_score: avg_quality
      formula.efficiency: avg_efficiency

  - type: table
    name: "⏰ 复习提醒"
    filters:
      or:
        - 'next_review && date(next_review) <= today() + "3d"'
        - 'next_review == ""'
    order:
      - formula.days_to_review
      - file.name
      - status
      - confidence
      - formula.review_overdue
```

### Daily Notes Index

```yaml
filters:
  and:
    - file.inFolder("Daily Notes")
    - '/^\d{4}-\d{2}-\d{2}$/.matches(file.basename)'

formulas:
  word_estimate: '(file.size / 5).round(0)'
  day_of_week: 'date(file.basename).format("dddd")'

properties:
  formula.day_of_week:
    displayName: "Day"
  formula.word_estimate:
    displayName: "~Words"

views:
  - type: table
    name: "Recent Notes"
    limit: 30
    order:
      - file.name
      - formula.day_of_week
      - formula.word_estimate
      - file.mtime
```

## Embedding Bases

Embed in Markdown files:

```markdown
![[MyBase.base]]

<!-- Specific view -->
![[MyBase.base#View Name]]
```

## YAML Quoting Rules

- Use single quotes for formulas containing double quotes: `'if(done, "Yes", "No")'`
- Use double quotes for simple strings: `"My View Name"`
- Escape nested quotes properly in complex expressions

## Common Patterns

### Filter by Tag
```yaml
filters:
  and:
    - file.hasTag("project")
```

### Filter by Folder
```yaml
filters:
  and:
    - file.inFolder("Notes")
```

### Filter by Date Range
```yaml
filters:
  and:
    - 'file.mtime > now() - "7d"'
```

### Filter by Property Value
```yaml
filters:
  and:
    - 'status == "active"'
    - 'priority >= 3'
```

### Combine Multiple Conditions
```yaml
filters:
  or:
    - and:
        - file.hasTag("important")
        - 'status != "done"'
    - and:
        - 'priority == 1'
        - 'due != ""'
```

## References

- [Bases Syntax](https://help.obsidian.md/bases/syntax)
- [Functions](https://help.obsidian.md/bases/functions)
- [Views](https://help.obsidian.md/bases/views)
- [Formulas](https://help.obsidian.md/formulas)

