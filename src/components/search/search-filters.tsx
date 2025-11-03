'use client'

import { useState } from 'react'
import { Icons } from '@/components/ui/icons'
import { type SearchFilters as SearchFiltersType } from '@/lib/search'

interface SearchFiltersProps {
  filters: SearchFiltersType
  onChange: (filters: SearchFiltersType) => void
  resultCount: number
}

export function SearchFilters({ filters, onChange, resultCount }: SearchFiltersProps) {
  const [expandedSections, setExpandedSections] = useState<string[]>(['tracks', 'difficulty'])

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
  }

  const handleFilterChange = (category: keyof SearchFiltersType, value: string, checked: boolean) => {
    const currentValues = filters[category] as string[]
    const newValues = checked 
      ? [...currentValues, value]
      : currentValues.filter(v => v !== value)
    
    onChange({
      ...filters,
      [category]: newValues
    })
  }

  const clearAllFilters = () => {
    onChange({
      tracks: [],
      difficulty: [],
      contentType: [],
      duration: []
    })
  }

  const hasActiveFilters = Object.values(filters).some(arr => arr.length > 0)

  const trackOptions = [
    { value: 'ai', label: 'Artificial Intelligence', icon: Icons.Bot, color: 'text-violet-600' },
    { value: 'data_engineer', label: 'Data Engineering', icon: Icons.Database, color: 'text-blue-600' },
    { value: 'saas', label: 'SaaS Development', icon: Icons.Cloud, color: 'text-purple-600' },
    { value: 'sfdc', label: 'Salesforce', icon: Icons.Zap, color: 'text-blue-500' },
    { value: 'snowflake_tune', label: 'Snowflake Tuning', icon: Icons.Snowflake, color: 'text-cyan-600' },
    { value: 'workato', label: 'Workato', icon: Icons.Link, color: 'text-teal-600' },
    { value: 'ba', label: 'Business Analyst', icon: Icons.BarChart3, color: 'text-green-600' },
    { value: 'data_gov', label: 'Data Governance', icon: Icons.Shield, color: 'text-emerald-600' },
    { value: 'devops_engineer', label: 'DevOps', icon: Icons.Settings, color: 'text-gray-600' },
    { value: 'finance', label: 'Finance', icon: Icons.DollarSign, color: 'text-green-500' },
    { value: 'hr', label: 'Human Resources', icon: Icons.Users, color: 'text-pink-600' },
    { value: 'qa', label: 'Quality Assurance', icon: Icons.CheckCircle, color: 'text-green-600' },
    { value: 'rpa', label: 'RPA', icon: Icons.Bot, color: 'text-orange-600' }
  ]

  const difficultyOptions = [
    { value: 'beginner', label: 'Beginner', description: 'No prior experience needed' },
    { value: 'intermediate', label: 'Intermediate', description: 'Some experience helpful' },
    { value: 'advanced', label: 'Advanced', description: 'Expert level content' }
  ]

  const contentTypeOptions = [
    { value: 'lesson', label: 'Lessons', icon: Icons.BookOpen },
    { value: 'module', label: 'Modules', icon: Icons.Book },
    { value: 'track', label: 'Learning Tracks', icon: Icons.GraduationCap }
  ]

  const durationOptions = [
    { value: 'short', label: 'Quick (< 30 min)', description: 'Short lessons' },
    { value: 'medium', label: 'Standard (30-60 min)', description: 'Normal length' },
    { value: 'long', label: 'Extended (> 1 hour)', description: 'In-depth content' }
  ]

  const FilterSection = ({ 
    title, 
    sectionKey, 
    children 
  }: { 
    title: string
    sectionKey: string
    children: React.ReactNode 
  }) => {
    const isExpanded = expandedSections.includes(sectionKey)
    
    return (
      <div className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
        <button
          onClick={() => toggleSection(sectionKey)}
          className="w-full flex items-center justify-between py-3 text-sm font-medium text-gray-900 dark:text-white hover:text-gray-700 dark:hover:text-gray-300"
        >
          <span>{title}</span>
          {isExpanded ? (
            <Icons.ChevronDown className="w-4 h-4" />
          ) : (
            <Icons.ChevronRight className="w-4 h-4" />
          )}
        </button>
        
        {isExpanded && (
          <div className="pb-4 space-y-2">
            {children}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-gray-900 dark:text-white">
          Filters
        </h3>
        {hasActiveFilters && (
          <button
            onClick={clearAllFilters}
            className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Results Count */}
      {resultCount > 0 && (
        <div className="mb-4 p-2 bg-gray-50 dark:bg-gray-750 rounded text-xs text-gray-600 dark:text-gray-400">
          {resultCount} result{resultCount !== 1 ? 's' : ''} found
        </div>
      )}

      {/* Filter Sections */}
      <div className="space-y-0">
        {/* Tracks */}
        <FilterSection title="Learning Tracks" sectionKey="tracks">
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {trackOptions.map((track) => {
              const IconComponent = track.icon
              return (
                <label key={track.value} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 p-1 rounded">
                  <input
                    type="checkbox"
                    checked={filters.tracks.includes(track.value)}
                    onChange={(e) => handleFilterChange('tracks', track.value, e.target.checked)}
                    className="w-3 h-3 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <IconComponent className={`w-3 h-3 ${track.color}`} />
                  <span className="text-xs text-gray-700 dark:text-gray-300 flex-1">
                    {track.label}
                  </span>
                </label>
              )
            })}
          </div>
        </FilterSection>

        {/* Difficulty */}
        <FilterSection title="Difficulty Level" sectionKey="difficulty">
          <div className="space-y-1">
            {difficultyOptions.map((level) => (
              <label key={level.value} className="flex items-start space-x-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 p-1 rounded">
                <input
                  type="checkbox"
                  checked={filters.difficulty.includes(level.value)}
                  onChange={(e) => handleFilterChange('difficulty', level.value, e.target.checked)}
                  className="w-3 h-3 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    {level.label}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {level.description}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </FilterSection>

        {/* Content Type */}
        <FilterSection title="Content Type" sectionKey="contentType">
          <div className="space-y-1">
            {contentTypeOptions.map((type) => {
              const IconComponent = type.icon
              return (
                <label key={type.value} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 p-1 rounded">
                  <input
                    type="checkbox"
                    checked={filters.contentType.includes(type.value)}
                    onChange={(e) => handleFilterChange('contentType', type.value, e.target.checked)}
                    className="w-3 h-3 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <IconComponent className="w-3 h-3 text-gray-500" />
                  <span className="text-xs text-gray-700 dark:text-gray-300">
                    {type.label}
                  </span>
                </label>
              )
            })}
          </div>
        </FilterSection>

        {/* Duration */}
        <FilterSection title="Duration" sectionKey="duration">
          <div className="space-y-1">
            {durationOptions.map((duration) => (
              <label key={duration.value} className="flex items-start space-x-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 p-1 rounded">
                <input
                  type="checkbox"
                  checked={filters.duration.includes(duration.value)}
                  onChange={(e) => handleFilterChange('duration', duration.value, e.target.checked)}
                  className="w-3 h-3 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-0.5"
                />
                <div className="flex-1">
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    {duration.label}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {duration.description}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </FilterSection>
      </div>
    </div>
  )
}