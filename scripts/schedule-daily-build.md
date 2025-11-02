# Daily Module Description Build Setup

This document explains how to set up the daily automated build for module descriptions.

## Quick Setup

### Option 1: Windows Task Scheduler (Recommended)

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create Basic Task**
   - Click "Create Basic Task..." in the right panel
   - Name: `LMS Module Description Daily Build`
   - Description: `Daily regeneration of module descriptions for all tracks`

3. **Set Trigger**
   - Trigger: Daily
   - Start time: `02:00 AM` (or preferred time)
   - Recur every: `1 days`

4. **Set Action** (Choose one option)
   
   **Option A: Batch Script**
   - Action: Start a program
   - Program/script: `C:\ai\training\lms-platform\scripts\daily-build.bat`
   - Start in: `C:\ai\training\lms-platform`
   
   **Option B: PowerShell Script (Recommended)**
   - Action: Start a program
   - Program/script: `powershell.exe`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\ai\training\lms-platform\scripts\daily-build.ps1"`
   - Start in: `C:\ai\training\lms-platform`

5. **Finish**
   - Check "Open the Properties dialog" and click Finish

6. **Configure Additional Settings**
   - In Properties dialog:
     - General tab: Check "Run whether user is logged on or not"
     - Settings tab: Check "Run task as soon as possible after a scheduled start is missed"

### Option 2: Manual Daily Run

**Command Line:**
```bash
cd C:\ai\training\lms-platform
"C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe" scripts/build_module_descriptions.py all
```

**Double-click options:**
- `scripts/daily-build.bat` (Batch version)
- `scripts/daily-build.ps1` (PowerShell version - recommended)

**PowerShell:**
```powershell
cd C:\ai\training\lms-platform
powershell -ExecutionPolicy Bypass -File "scripts/daily-build.ps1"
```

## What the Daily Build Does

1. **Scans all tracks** in `src/data/` directory
2. **Analyzes lesson files** in each track's `lessons/` folder  
3. **Generates comprehensive metadata** for each track including:
   - **Track information**: type, title, description, duration, statistics
   - **Module descriptions**: prerequisites, skills gained, key topics, progress tracking
   - **Detailed lesson metadata**: learning objectives, complexity, duration, tags, navigation
   - **Lesson content analysis**: code examples count, exercise detection, lesson types
   - **Navigation structure**: next/previous relationships, module ordering, lesson maps
4. **Saves to** `src/data/{track}/modules-descriptions/module.json`
5. **Provides summary report** of successful/failed builds with duration statistics

## Tracks Processed

The daily build will process all available tracks:

**Book of Knowledge (6 tracks):**
- AI, Data Engineering, SaaS, Salesforce, Snowflake Tuning, Workato

**Learning Paths (16 tracks):**
- Business Analyst, Data Engineer, Data Governance, DevOps Engineer, Finance, Human Resources, Marketing, Master Data Management, Project Manager, Quality Assurance, RPA, Sales, Salesforce Engineer, Talent Acquisition, Visualization Engineer, Workato Engineer

**Total: 22 tracks** with comprehensive lesson-level metadata

## Monitoring

- **Build logs** are saved to `scripts/build-log.txt`
- **Console output** shows detailed progress for each track
- **Exit codes**: 0 = success, 1 = failure

## Troubleshooting

If the daily build fails:

1. Check that Python path is correct: `C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe`
2. Verify all track folders exist in `src/data/`
3. Ensure lesson files exist in each track's `lessons/` folder
4. Check `scripts/build-log.txt` for error details

## Manual Testing

Test the daily build manually:

```bash
# Test single track
python scripts/build_module_descriptions.py ai

# Test all tracks
python scripts/build_module_descriptions.py all
```

## Enhanced Features (Latest Update)

The daily build now generates **comprehensive lesson-level metadata** for dynamic website rendering:

### Lesson-Level Data
- ✅ **Learning objectives** extracted from lesson content
- ✅ **Estimated duration** (reading time + exercises + code review)
- ✅ **Complexity levels** (Beginner/Intermediate/Advanced)
- ✅ **Lesson types** (Theory/Hands-on/Assessment/Practical/Overview)
- ✅ **Technical tags** (API, Database, Cloud, Security, etc.)
- ✅ **Code examples count** and exercise detection
- ✅ **Key concepts** and prerequisites
- ✅ **URL-friendly slugs** for routing

### Navigation & Progress
- ✅ **Next/Previous relationships** for seamless lesson progression  
- ✅ **Module ordering** and lesson maps for navigation
- ✅ **Progress tracking structure** ready for implementation
- ✅ **Search/filter metadata** (tags, complexity, type, duration)

### Statistics & Analytics
- ✅ **Track-level statistics** (total duration, practical ratio, lesson types)
- ✅ **Module-level aggregations** (hands-on lessons, estimated time)
- ✅ **Content analysis** (average lesson duration, complexity distribution)

## Benefits of Daily Builds

- ✅ **Always up-to-date** module descriptions with lesson-level metadata
- ✅ **Automatic synchronization** when lessons are added/modified  
- ✅ **Consistent data structure** across all 22 tracks
- ✅ **Error detection** for malformed lesson files
- ✅ **Website-ready JSON** for dynamic page rendering
- ✅ **Zero maintenance** once scheduled