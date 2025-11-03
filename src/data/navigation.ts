import { MenuItem } from '@/types/navigation'

export const navigationData: MenuItem[] = [
  {
    id: 'learning-content',
    label: 'Learning Content',
    icon: 'Book',
    href: '/learning-content',
    status: 'active'
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