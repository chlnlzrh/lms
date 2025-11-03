import { MenuItem } from '@/types/navigation'

export const navigationData: MenuItem[] = [
  {
    id: 'book-of-knowledge',
    label: 'Book of Knowledge',
    icon: 'BookOpen',
    href: '/book-of-knowledge',
    status: 'active',
    children: [
      {
        id: 'ai',
        label: 'Artificial Intelligence',
        icon: 'Bot',
        href: '/book-of-knowledge/ai',
        status: 'active'
      },
      {
        id: 'de',
        label: 'Data Engineering',
        icon: 'Database',
        href: '/book-of-knowledge/de',
        status: 'active'
      },
      {
        id: 'saas',
        label: 'SaaS',
        icon: 'Cloud',
        href: '/book-of-knowledge/saas',
        status: 'active'
      },
      {
        id: 'sfdc',
        label: 'Salesforce',
        icon: 'Zap',
        href: '/book-of-knowledge/sfdc',
        status: 'active'
      },
      {
        id: 'snowflake_tune',
        label: 'Snowflake Tuning',
        icon: 'Snowflake',
        href: '/book-of-knowledge/snowflake_tune',
        status: 'active'
      },
      {
        id: 'workato',
        label: 'Workato',
        icon: 'Link',
        href: '/book-of-knowledge/workato',
        status: 'active'
      }
    ]
  },
  {
    id: 'learning-path',
    label: 'Learning Path',
    icon: 'GraduationCap',
    href: '/learning-path',
    status: 'active',
    children: [
      {
        id: 'ba',
        label: 'Business Analyst',
        icon: 'BarChart3',
        href: '/learning-path/ba',
        status: 'active'
      },
      {
        id: 'data_engineer',
        label: 'Data Engineer',
        icon: 'Database',
        href: '/learning-path/data_engineer',
        status: 'active'
      },
      {
        id: 'data_gov',
        label: 'Data Governance',
        icon: 'Shield',
        href: '/learning-path/data_gov',
        status: 'active'
      },
      {
        id: 'devops_engineer',
        label: 'DevOps Engineer',
        icon: 'Settings',
        href: '/learning-path/devops_engineer',
        status: 'active'
      },
      {
        id: 'finance',
        label: 'Finance',
        icon: 'DollarSign',
        href: '/learning-path/finance',
        status: 'active'
      },
      {
        id: 'hr',
        label: 'Human Resources',
        icon: 'Users',
        href: '/learning-path/hr',
        status: 'active'
      },
      {
        id: 'marketing',
        label: 'Marketing',
        icon: 'Megaphone',
        href: '/learning-path/marketing',
        status: 'active'
      },
      {
        id: 'mdm',
        label: 'Master Data Management',
        icon: 'Archive',
        href: '/learning-path/mdm',
        status: 'active'
      },
      {
        id: 'pm',
        label: 'Project Manager',
        icon: 'Briefcase',
        href: '/learning-path/pm',
        status: 'active'
      },
      {
        id: 'qa',
        label: 'Quality Assurance',
        icon: 'CheckCircle',
        href: '/learning-path/qa',
        status: 'active'
      },
      {
        id: 'rpa',
        label: 'Robotic Process Automation',
        icon: 'Bot',
        href: '/learning-path/rpa',
        status: 'active'
      },
      {
        id: 'sales',
        label: 'Sales',
        icon: 'TrendingUp',
        href: '/learning-path/sales',
        status: 'active'
      },
      {
        id: 'sfdc_engineer',
        label: 'Salesforce Engineer',
        icon: 'Zap',
        href: '/learning-path/sfdc_engineer',
        status: 'active'
      },
      {
        id: 'ta',
        label: 'Talent Acquisition',
        icon: 'FileSearch',
        href: '/learning-path/ta',
        status: 'active'
      },
      {
        id: 'viz_engineer',
        label: 'Visualization Engineer',
        icon: 'PieChart',
        href: '/learning-path/viz_engineer',
        status: 'active'
      },
      {
        id: 'workato_engineer',
        label: 'Workato Engineer',
        icon: 'Link',
        href: '/learning-path/workato_engineer',
        status: 'active'
      }
    ]
  },
  {
    id: 'employee-onboarding',
    label: 'Employee Onboarding',
    icon: 'ClipboardList',
    status: 'coming-soon'
  },
  {
    id: 'compliance-training',
    label: 'Compliance Training',
    icon: 'ScrollText',
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