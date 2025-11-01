export interface MenuItem {
  id: string
  label: string
  icon: string
  href?: string
  status: 'active' | 'coming-soon'
  children?: MenuItem[]
  lessonCount?: number
}

export interface NavigationState {
  expandedSections: string[]
  currentPath: string
}