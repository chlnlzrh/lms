# Module Description Builder

A general-purpose tool to automatically generate comprehensive module descriptions for any learning track by analyzing lesson content and structure.

## Usage

```bash
python scripts/build_module_descriptions.py <track_folder_name>
```

## Available Tracks

| Track | Description | Status |
|-------|-------------|--------|
| `mdm` | Master Data Management | ✅ Ready |
| `rpa` | Robotic Process Automation | ✅ Ready |
| `workato` | Integration Platform | ✅ Ready |
| `snowflake_tune` | Snowflake Performance | ✅ Ready |
| `sfdc` | Salesforce Development | ✅ Ready |
| `data_gov` | Data Governance | ✅ Ready |

## Examples

Generate module descriptions for MDM track:
```bash
python scripts/build_module_descriptions.py mdm
```

Generate module descriptions for RPA track:
```bash
python scripts/build_module_descriptions.py rpa
```

Generate module descriptions for Workato track:
```bash
python scripts/build_module_descriptions.py workato
```

## Output

The script creates a comprehensive JSON file at:
```
src/data/{track_name}/modules-descriptions/module.json
```

## Features

✅ **Automatic Lesson Analysis** - Analyzes all lesson files and frontmatter
✅ **Intelligent Grouping** - Groups lessons into logical modules (24 lessons per module)
✅ **Content Extraction** - Extracts topics, skills, and learning objectives
✅ **Duration Estimation** - Calculates realistic time estimates based on complexity
✅ **Progressive Structure** - Creates progressive learning paths from foundation to advanced
✅ **Track-Specific Themes** - Adapts content to each track's focus and terminology

## Generated Structure

Each module includes:

- **ID & Title** - Unique identifier and descriptive title
- **Subtitle** - Brief description of module focus
- **Duration** - Estimated completion time in weeks
- **Lesson/Lab Count** - Number of lessons and hands-on labs
- **Prerequisites** - Required knowledge and previous modules
- **Key Topics** - Main technical topics covered
- **Skills Gained** - Specific competencies developed

## Requirements

- Python 3.11+
- python-frontmatter library (`pip install python-frontmatter`)

## Track Configurations

The script includes specialized configurations for each track:

| Track | Theme | Color | Icon | Focus Areas |
|-------|-------|-------|------|-------------|
| MDM | Master Data Management | Indigo | Database | Data integration, governance, quality |
| RPA | Robotic Process Automation | Orange | Bot | Automation, workflows, efficiency |
| Workato | Integration Platform | Teal | Link | API integration, workflows, connectivity |
| Snowflake | Performance Tuning | Cyan | Zap | Performance optimization, tuning, scaling |
| SFDC | Salesforce Development | Blue | Cloud | CRM, platform development, customization |
| Data Gov | Data Governance | Emerald | Shield | Governance, compliance, stewardship |

## Integration

The generated module descriptions are compatible with the existing LMS platform and follow the same structure as AI, Data Engineering, and SaaS tracks.