'use client'

import { ParsedLesson } from '@/types/content'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'
import { progressTracker } from '@/lib/progress-tracker'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useState, useEffect } from 'react'

interface LessonContentProps {
  lesson: ParsedLesson
  previousLesson?: ParsedLesson
  nextLesson?: ParsedLesson
  onMarkComplete?: () => void
  isCompleted?: boolean
  className?: string
}

export function LessonContent({ 
  lesson, 
  previousLesson, 
  nextLesson, 
  onMarkComplete,
  isCompleted = false,
  className 
}: LessonContentProps) {
  const [completed, setCompleted] = useState(isCompleted)
  const [bookmarked, setBookmarked] = useState(false)
  const [timeSpent, setTimeSpent] = useState(0)

  const trackColors = {
    ai: 'blue',
    'data-engineering': 'green'
  }

  const trackIcons = {
    ai: Icons.Bot,
    'data-engineering': Icons.Database
  }

  const color = trackColors[lesson.track]
  const IconComponent = trackIcons[lesson.track]

  // Load progress data on mount
  useEffect(() => {
    const progress = progressTracker.getLessonProgress(lesson.id)
    if (progress) {
      setCompleted(progress.isCompleted)
      setBookmarked(progress.bookmarked)
      setTimeSpent(progress.timeSpent)
    }

    // Mark lesson as accessed
    progressTracker.markLessonAccessed(lesson)

    // Track time spent (simple implementation)
    const startTime = Date.now()
    return () => {
      const sessionTime = Math.round((Date.now() - startTime) / 60000) // Convert to minutes
      if (sessionTime > 0) {
        const currentProgress = progressTracker.getLessonProgress(lesson.id)
        if (currentProgress) {
          currentProgress.timeSpent += sessionTime
          setTimeSpent(currentProgress.timeSpent)
        }
      }
    }
  }, [lesson])

  const handleMarkComplete = () => {
    progressTracker.markLessonComplete(lesson, timeSpent)
    setCompleted(true)
    if (onMarkComplete) {
      onMarkComplete()
    }
  }

  const handleToggleBookmark = () => {
    const newBookmarkState = progressTracker.toggleBookmark(lesson.id)
    setBookmarked(newBookmarkState)
  }

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Talent Development', href: '/talent-development' },
    { 
      label: lesson.track === 'ai' ? 'AI Training' : 'Data Engineering', 
      href: `/talent-development/${lesson.track}` 
    },
    { 
      label: lesson.frontmatter.module || `Module ${lesson.moduleNumber}`,
      href: `/talent-development/${lesson.track}/module-${lesson.moduleNumber}`
    },
    { label: lesson.frontmatter.title }
  ]

  // Process content to highlight code blocks
  const processContent = (content: string) => {
    // Split content by code blocks
    const parts = content.split(/(```[\s\S]*?```)/g)
    
    return parts.map((part, index) => {
      if (part.startsWith('```')) {
        // Extract language and code
        const lines = part.split('\n')
        const language = lines[0].replace('```', '').trim() || 'text'
        const code = lines.slice(1, -1).join('\n')
        
        return (
          <div key={index} className="my-4">
            <SyntaxHighlighter
              language={language}
              style={tomorrow}
              customStyle={{
                background: '#1a1a1a',
                borderRadius: '8px',
                fontSize: '12px',
                lineHeight: '1.4'
              }}
            >
              {code}
            </SyntaxHighlighter>
          </div>
        )
      } else {
        // Regular markdown content
        return (
          <div 
            key={index}
            className="prose prose-sm max-w-none dark:prose-invert"
            dangerouslySetInnerHTML={{ __html: part }}
          />
        )
      }
    })
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Breadcrumbs */}
      <Breadcrumbs items={breadcrumbItems} />

      {/* Lesson Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-start space-x-3 flex-1">
            <IconComponent className={`w-6 h-6 text-${color}-600 mt-1 flex-shrink-0`} />
            
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <span className={`text-xs px-2 py-1 rounded bg-${color}-100 text-${color}-700 dark:bg-${color}-900 dark:text-${color}-300`}>
                  {lesson.frontmatter.module || `Module ${lesson.moduleNumber}`}
                </span>
                {lesson.frontmatter.complexity && (
                  <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                    Level: {lesson.frontmatter.complexity === 'F' ? 'Foundation' : 
                            lesson.frontmatter.complexity === 'I' ? 'Intermediate' : 'Advanced'}
                  </span>
                )}
                {completed && (
                  <span className="text-xs px-2 py-1 rounded bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 flex items-center space-x-1">
                    <Icons.Circle className="w-3 h-3 fill-current" />
                    <span>Completed</span>
                  </span>
                )}
              </div>
              
              <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                {lesson.frontmatter.title}
              </h1>
              
              <div className="flex items-center space-x-4 text-xs text-gray-600 dark:text-gray-300">
                <div className="flex items-center space-x-1">
                  <Icons.Clock className="w-3 h-3" />
                  <span>{lesson.estimatedReadTime} min read</span>
                </div>
                {lesson.frontmatter.duration && (
                  <div className="flex items-center space-x-1">
                    <Icons.Calendar className="w-3 h-3" />
                    <span>{lesson.frontmatter.duration}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={handleToggleBookmark}
              className={cn(
                "p-2 rounded transition-colors",
                bookmarked 
                  ? "text-yellow-600 hover:text-yellow-700" 
                  : "text-gray-400 hover:text-gray-600"
              )}
              title={bookmarked ? "Remove bookmark" : "Add bookmark"}
            >
              <Icons.Bookmark className={cn("w-4 h-4", bookmarked && "fill-current")} />
            </button>
            
            {!completed && (
              <button
                onClick={handleMarkComplete}
                className={`bg-${color}-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-${color}-700 transition-colors`}
              >
                Mark Complete
              </button>
            )}
          </div>
        </div>

        {/* Learning Objectives */}
        {lesson.frontmatter.learningObjectives && lesson.frontmatter.learningObjectives.length > 0 && (
          <div className="border-t border-gray-100 dark:border-gray-700 pt-4">
            <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-2">
              Learning Objectives
            </h3>
            <ul className="text-xs text-gray-600 dark:text-gray-300 space-y-1">
              {lesson.frontmatter.learningObjectives.map((objective, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <span className="text-gray-400 mt-0.5">•</span>
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Lesson Content */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="prose prose-sm max-w-none dark:prose-invert">
          {processContent(lesson.content)}
        </div>
      </div>

      {/* Prerequisites */}
      {lesson.frontmatter.prerequisites && lesson.frontmatter.prerequisites.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <h3 className="text-xs font-bold text-yellow-800 dark:text-yellow-200 mb-2">
            Prerequisites
          </h3>
          <ul className="text-xs text-yellow-700 dark:text-yellow-300 space-y-1">
            {lesson.frontmatter.prerequisites.map((prereq, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-yellow-500 mt-0.5">•</span>
                <span>{prereq}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center justify-between">
        {previousLesson ? (
          <a
            href={`/talent-development/${lesson.track}/lesson/${previousLesson.slug}`}
            className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <Icons.ChevronLeft className="w-4 h-4" />
            <div>
              <div className="text-gray-500">Previous</div>
              <div className="font-normal">{previousLesson.frontmatter.title}</div>
            </div>
          </a>
        ) : (
          <div></div>
        )}

        {nextLesson ? (
          <a
            href={`/talent-development/${lesson.track}/lesson/${nextLesson.slug}`}
            className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors text-right"
          >
            <div>
              <div className="text-gray-500">Next</div>
              <div className="font-normal">{nextLesson.frontmatter.title}</div>
            </div>
            <Icons.ChevronRight className="w-4 h-4" />
          </a>
        ) : (
          <div></div>
        )}
      </div>
    </div>
  )
}