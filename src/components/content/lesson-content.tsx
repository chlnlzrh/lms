'use client'

import { ParsedLesson } from '@/types/content'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'
import { progressTracker } from '@/lib/progress-tracker'
import { CodeBlock } from '@/components/ui/code-block'
import { useState, useEffect } from 'react'
import React from 'react'

interface LessonContentProps {
  lesson: ParsedLesson
  previousLesson?: ParsedLesson
  nextLesson?: ParsedLesson
  isCompleted?: boolean
  className?: string
}

export function LessonContent({ 
  lesson, 
  previousLesson, 
  nextLesson, 
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

  // Add syntax highlighting CSS after mount
  useEffect(() => {
    // Add enhanced syntax highlighting styles
    const addSyntaxHighlightingCSS = () => {
      if (document.getElementById('syntax-highlighting-styles')) return
      
      const style = document.createElement('style')
      style.id = 'syntax-highlighting-styles'
      style.textContent = `
        .language-python .token.comment,
        .language-python .token.prolog,
        .language-python .token.doctype,
        .language-python .token.cdata {
          color: #8e9aaf !important;
        }
        .language-python .token.punctuation {
          color: #d6deeb !important;
        }
        .language-python .token.property,
        .language-python .token.tag,
        .language-python .token.boolean,
        .language-python .token.number,
        .language-python .token.constant,
        .language-python .token.symbol,
        .language-python .token.deleted {
          color: #addb67 !important;
        }
        .language-python .token.selector,
        .language-python .token.attr-name,
        .language-python .token.string,
        .language-python .token.char,
        .language-python .token.builtin,
        .language-python .token.inserted {
          color: #ecc48d !important;
        }
        .language-python .token.operator,
        .language-python .token.entity,
        .language-python .token.url {
          color: #7fdbca !important;
        }
        .language-python .token.atrule,
        .language-python .token.attr-value,
        .language-python .token.keyword {
          color: #c792ea !important;
        }
        .language-python .token.function,
        .language-python .token.class-name {
          color: #82aaff !important;
        }
        .language-python .token.regex,
        .language-python .token.important,
        .language-python .token.variable {
          color: #d6deeb !important;
        }
        .language-python .token.important,
        .language-python .token.bold {
          font-weight: bold !important;
        }
        .language-python .token.italic {
          font-style: italic !important;
        }
      `
      document.head.appendChild(style)
    }

    // Run Prism highlighting on code blocks
    const highlightCodeBlocks = () => {
      if (typeof window !== 'undefined' && (window as any).Prism) {
        (window as any).Prism.highlightAll()
      } else {
        // Load Prism if not available
        import('prismjs').then((Prism) => {
          import('prismjs/components/prism-python').then(() => {
            Prism.highlightAll()
          })
        })
      }
    }

    addSyntaxHighlightingCSS()
    highlightCodeBlocks()
  }, [])

  const handleMarkComplete = () => {
    progressTracker.markLessonComplete(lesson, timeSpent)
    setCompleted(true)
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

  // Process content - simple HTML rendering for stability
  const processContent = (htmlContent: string) => {
    // Always render enhanced HTML with consistent styling
    const enhancedContent = htmlContent.replace(
      /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
      (match, language, code) => {
        return `<div class="my-4 relative">
          <div class="bg-gray-800 text-gray-200 rounded-t-lg px-3 py-1 text-xs font-mono">
            ${language || 'code'}
          </div>
          <pre class="bg-gray-900 text-gray-100 rounded-b-lg p-4 overflow-x-auto"><code class="language-${language}">${code}</code></pre>
        </div>`
      }
    )
    
    return (
      <div 
        className="prose prose-sm max-w-none dark:prose-invert"
        dangerouslySetInnerHTML={{ __html: enhancedContent }}
      />
    )
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
          {processContent(lesson.htmlContent)}
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