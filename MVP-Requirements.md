# LMS MVP Requirements

## Project Overview
Desktop-only Learning Management System with complete menu structure showing current capabilities and future roadmap, built with Next.js 14 + TypeScript following CLAUDE.md standards.

## Complete Navigation Structure (MVP Implementation)

### Primary Menu (Left Sidebar)
```
🏠 Dashboard
├── 📋 Employee Onboarding (Coming Soon)
├── 📚 Talent Development
│   ├── 🤖 AI Training Track (ACTIVE - 234 lessons)
│   │   ├── Module 1: AI Foundation & Tool Fluency (66 lessons)
│   │   ├── Module 2: AI in SDLC (32 lessons)
│   │   ├── Module 3: AI-Augmented Engineering (48 lessons)
│   │   ├── Module 4: AI Agent & Platform Architecture (42 lessons)
│   │   ├── Module 5: AI Strategy & Governance (23 lessons)
│   │   └── Module 6: Continuous Learning & Innovation (23 lessons)
│   ├── 🗃️ Data Engineering Track (ACTIVE - 300+ lessons)
│   │   ├── Module 1: Database Fundamentals
│   │   ├── Module 2: SQL & ELT Concepts
│   │   ├── Module 3: Data Warehousing Principles
│   │   ├── Module 4: Data Modeling
│   │   ├── Module 5: Snowflake Specific Knowledge
│   │   ├── Modules 6-15: [Various DE specializations]
│   │   └── Module 20: Emerging Topics & Advanced Concepts
│   ├── 🔗 Integration Track (Coming Soon)
│   ├── ⚙️ SaaS App Build Track (Coming Soon)
│   ├── ☁️ Salesforce Track (Coming Soon)
│   └── 🛡️ MDM & Data Governance Track (Coming Soon)
├── 📜 Compliance Training (Coming Soon)
├── 💼 Sales Enablement (Coming Soon)
├── 🎓 Customer Education (Coming Soon)
├── 📅 Calendar/Events (Coming Soon)
├── 📖 Learning Catalog (ACTIVE)
├── 🔍 Search (ACTIVE)
├── 👤 Profile/My Account (ACTIVE)
├── 💬 Support/Help (ACTIVE)
└── ⚙️ Administration (Coming Soon)
```

## MVP Active Features

### 1. Dashboard (🏠) - ACTIVE
**Current Implementation:**
- Welcome screen with user name display
- Continue learning section (resume last accessed lesson)
- Progress overview for AI and DE tracks (visual progress bars)
- Quick stats: total lessons completed, current module
- Recent activity feed (last 5 accessed lessons)
- Future roadmap preview cards

### 2. Talent Development (📚) - PARTIALLY ACTIVE

#### AI Training Track (🤖) - FULLY ACTIVE
**Content Source:** `/ai/lessons/` directory (234 markdown files)
**Features:**
- 6 modules with collapsible navigation
- Sequential lesson progression (M01-L001 naming convention)
- Individual lesson display with markdown rendering
- Progress tracking per lesson and module
- Previous/Next navigation within modules

#### Data Engineering Track (🗃️) - FULLY ACTIVE
**Content Source:** `/de/lessons/` directory (300+ markdown files)
**Features:**
- 20 modules with collapsible navigation
- Sequential lesson progression
- Individual lesson display with markdown rendering
- Progress tracking per lesson and module
- Previous/Next navigation within modules

#### Future Tracks - PREVIEW MODE
- **Integration Track:** Shows "Coming Soon" with planned modules
- **SaaS App Build Track:** Shows "Coming Soon" with planned modules
- **Salesforce Track:** Shows "Coming Soon" with planned modules
- **MDM & Data Governance Track:** Shows "Coming Soon" with planned modules

### 3. Learning Catalog (📖) - ACTIVE
**Features:**
- Global view of all available content
- Filter by track (AI/DE)
- Search functionality
- Lesson previews and metadata
- Bookmark system

### 4. Search (🔍) - ACTIVE
**Features:**
- Global search across all lesson titles
- Basic keyword search in lesson content
- Search results with lesson preview
- Filter by track and module
- Quick search from header bar

### 5. Profile/My Account (👤) - ACTIVE
**Features:**
- User name and role display
- Learning progress overview
- Completed lessons list
- Current learning streaks
- Bookmark management for favorite lessons
- Achievement preview (badges coming soon)

### 6. Support/Help (💬) - ACTIVE
**Features:**
- Getting started guide
- Platform navigation tutorial
- FAQ section
- Contact information
- Feature request form

## Coming Soon Features (Visible but Disabled)

### 7. Employee Onboarding (📋) - COMING SOON
**Preview Description:**
- Role assessment and track assignment
- Welcome orientation program
- Environment setup guides
- Mentor assignment system

### 8. Compliance Training (📜) - COMING SOON
**Preview Description:**
- Mandatory course tracking
- Regulatory compliance modules
- Completion certificates
- Deadline management

### 9. Sales Enablement (💼) - COMING SOON
**Preview Description:**
- Product demonstration training
- Client presentation materials
- Competitive analysis modules
- Demo environment access

### 10. Customer Education (🎓) - COMING SOON
**Preview Description:**
- External-facing course catalog
- Client onboarding materials
- Partner certification programs
- Self-service learning portal

### 11. Calendar/Events (📅) - COMING SOON
**Preview Description:**
- Live session scheduling
- Assignment deadlines
- Learning events calendar
- Automated reminders

### 12. Administration (⚙️) - COMING SOON
**Preview Description:**
- User management
- Content creation tools
- Analytics dashboard
- System configuration

## UI Implementation Strategy

### Navigation States
1. **ACTIVE:** Full functionality, normal styling
2. **COMING SOON:** Visible with disabled state, tooltip explanation
3. **PREVIEW:** Clickable to show detailed roadmap information

### Visual Indicators
- **Green dot:** Active features
- **Orange dot:** Coming soon features
- **Lesson counts:** Show actual numbers for active tracks
- **Progress bars:** Only for active content
- **Tooltips:** Explain coming soon features

### Menu Interaction
```css
.menu-item-active {
  @apply text-black dark:text-white cursor-pointer;
}

.menu-item-coming-soon {
  @apply text-gray-400 cursor-not-allowed relative;
}

.menu-item-coming-soon::after {
  content: "Coming Soon";
  @apply text-xs bg-orange-100 text-orange-600 px-2 py-1 rounded ml-2;
}
```

## Technical Requirements

### Tech Stack
- **Framework:** Next.js 14 App Router
- **Language:** TypeScript (strict mode)
- **Styling:** Tailwind CSS + shadcn/ui components
- **Content:** File-based markdown parsing for active tracks
- **State Management:** Local storage for progress tracking
- **Deployment:** Vercel

### Content Management

#### Active Content Structure
```
/content/
├── talent-development/
│   ├── ai/ (ACTIVE)
│   │   ├── lessons/ (234 .md files)
│   │   ├── modules-descriptions/
│   │   └── Content Structure.md
│   └── data-engineering/ (ACTIVE)
│       ├── lessons/ (300+ .md files)
│       ├── modules-descriptions/
│       └── Content Structure.md
```

#### Future Content Placeholders
```
/content/
├── employee-onboarding/ (placeholder)
├── talent-development/
│   ├── integration/ (placeholder)
│   ├── saas-app-build/ (placeholder)
│   ├── salesforce/ (placeholder)
│   └── mdm-governance/ (placeholder)
├── compliance/ (placeholder)
├── sales-enablement/ (placeholder)
└── customer-education/ (placeholder)
```

## UI/UX Requirements (CLAUDE.md Compliant)

### Typography
- **All text:** Inter font, `text-xs font-normal`
- **Headers only:** `text-xs font-bold`
- **Menu spacing:** `py-1` to `py-1.5` max, sections `space-y-0.5`
- **Status indicators:** Small badges for feature states

### Navigation Design
- **Desktop sidebar:** Collapsed by default (icon-only)
- **Expand on hover/click:** 300ms spring animation
- **Active items:** `text-black dark:text-white`
- **Coming soon items:** `text-gray-400` with tooltips
- **Progressive disclosure:** Show full roadmap structure

## Customer Value Proposition

### Immediate Value (MVP)
- **500+ lessons** across AI and Data Engineering
- **Complete learning platform** with progress tracking
- **Professional development** in cutting-edge technologies
- **Structured curriculum** from foundation to advanced topics

### Future Value (Roadmap)
- **Complete enterprise LMS** covering all business functions
- **6 specialized tracks** for comprehensive skill development
- **Compliance and certification** management
- **Sales and customer education** capabilities
- **Advanced administration** and analytics

## Success Criteria

### MVP Launch
1. ✅ Complete menu structure visible (builds anticipation)
2. ✅ AI and DE tracks fully functional (immediate value)
3. ✅ Professional user experience (credibility)
4. ✅ Clear roadmap communication (future vision)
5. ✅ Smooth performance and navigation

### Customer Satisfaction
1. ✅ Users understand full platform potential
2. ✅ Active tracks provide immediate learning value
3. ✅ Coming soon features generate excitement
4. ✅ Professional design builds confidence
5. ✅ Clear development roadmap shown

This approach delivers immediate value while showcasing the complete vision, helping customers understand both current capabilities and future potential.