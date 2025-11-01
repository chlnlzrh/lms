import { MenuItem } from '@/types/navigation'

export const navigationData: MenuItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: 'Home',
    href: '/',
    status: 'active'
  },
  {
    id: 'employee-onboarding',
    label: 'Employee Onboarding',
    icon: 'ClipboardList',
    status: 'coming-soon'
  },
  {
    id: 'talent-development',
    label: 'Talent Development',
    icon: 'BookOpen',
    href: '/talent-development',
    status: 'active',
    children: [
      {
        id: 'ai-training',
        label: 'AI Training',
        icon: 'Bot',
        href: '/talent-development/ai',
        status: 'active',
        lessonCount: 234,
        children: [
          {
            id: 'ai-module-1',
            label: 'Module 1: AI Foundation & Tool Fluency',
            icon: 'Circle',
            href: '/talent-development/ai/module-1',
            status: 'active',
            lessonCount: 66
          },
          {
            id: 'ai-module-2',
            label: 'Module 2: AI in SDLC',
            icon: 'Circle',
            href: '/talent-development/ai/module-2',
            status: 'active',
            lessonCount: 32
          },
          {
            id: 'ai-module-3',
            label: 'Module 3: AI-Augmented Engineering',
            icon: 'Circle',
            href: '/talent-development/ai/module-3',
            status: 'active',
            lessonCount: 48
          },
          {
            id: 'ai-module-4',
            label: 'Module 4: AI Agent & Platform Architecture',
            icon: 'Circle',
            href: '/talent-development/ai/module-4',
            status: 'active',
            lessonCount: 42
          },
          {
            id: 'ai-module-5',
            label: 'Module 5: AI Strategy & Governance',
            icon: 'Circle',
            href: '/talent-development/ai/module-5',
            status: 'active',
            lessonCount: 23
          },
          {
            id: 'ai-module-6',
            label: 'Module 6: Continuous Learning & Innovation',
            icon: 'Circle',
            href: '/talent-development/ai/module-6',
            status: 'active',
            lessonCount: 23
          }
        ]
      },
      {
        id: 'data-engineering',
        label: 'Data Engineering',
        icon: 'Database',
        href: '/talent-development/data-engineering',
        status: 'active',
        lessonCount: 300,
        children: [
          {
            id: 'de-module-1',
            label: 'Module 1: Database Fundamentals',
            icon: 'Circle',
            href: '/talent-development/data-engineering/module-1',
            status: 'active'
          },
          {
            id: 'de-module-2',
            label: 'Module 2: SQL & ELT Concepts',
            icon: 'Circle',
            href: '/talent-development/data-engineering/module-2',
            status: 'active'
          },
          {
            id: 'de-module-3',
            label: 'Module 3: Data Warehousing Principles',
            icon: 'Circle',
            href: '/talent-development/data-engineering/module-3',
            status: 'active'
          },
          {
            id: 'de-module-4',
            label: 'Module 4: Data Modeling',
            icon: 'Circle',
            href: '/talent-development/data-engineering/module-4',
            status: 'active'
          },
          {
            id: 'de-module-5',
            label: 'Module 5: Snowflake Specific Knowledge',
            icon: 'Circle',
            href: '/talent-development/data-engineering/module-5',
            status: 'active'
          },
          {
            id: 'de-module-20',
            label: 'Module 20: Emerging Topics & Advanced Concepts',
            icon: 'Circle',
            href: '/talent-development/data-engineering/module-20',
            status: 'active'
          }
        ]
      },
      {
        id: 'integration',
        label: 'Integration',
        icon: 'Link',
        status: 'coming-soon'
      },
      {
        id: 'saas-app-build',
        label: 'SaaS App Build',
        icon: 'Settings',
        status: 'coming-soon'
      },
      {
        id: 'salesforce',
        label: 'Salesforce',
        icon: 'Cloud',
        status: 'coming-soon'
      },
      {
        id: 'mdm-governance',
        label: 'MDM & Data Governance',
        icon: 'Shield',
        status: 'coming-soon'
      }
    ]
  },
  {
    id: 'compliance-training',
    label: 'Compliance Training',
    icon: 'ScrollText',
    status: 'coming-soon'
  },
  {
    id: 'sales-enablement',
    label: 'Sales Enablement',
    icon: 'Briefcase',
    status: 'coming-soon'
  },
  {
    id: 'customer-education',
    label: 'Customer Education',
    icon: 'GraduationCap',
    status: 'coming-soon'
  },
  {
    id: 'calendar-events',
    label: 'Calendar/Events',
    icon: 'Calendar',
    status: 'coming-soon'
  },
  {
    id: 'learning-catalog',
    label: 'Learning Catalog',
    icon: 'Book',
    href: '/catalog',
    status: 'active'
  },
  {
    id: 'search',
    label: 'Search',
    icon: 'Search',
    href: '/search',
    status: 'active'
  },
  {
    id: 'profile',
    label: 'Profile/My Account',
    icon: 'User',
    href: '/profile',
    status: 'active'
  },
  {
    id: 'support',
    label: 'Support/Help',
    icon: 'MessageCircle',
    href: '/support',
    status: 'active'
  },
  {
    id: 'administration',
    label: 'Administration',
    icon: 'Settings',
    status: 'coming-soon'
  }
]