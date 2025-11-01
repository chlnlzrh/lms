import { notFound } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { LessonContent } from '@/components/content/lesson-content'
import { progressTracker } from '@/lib/progress-tracker'
import { contentParser } from '@/lib/content-parser'
import { Suspense } from 'react'

interface LessonPageProps {
  params: Promise<{
    track: string
    slug: string
  }>
}

async function LessonPageContent({ params }: LessonPageProps) {
  const resolvedParams = await params
  const { track, slug } = resolvedParams
  
  // Get all lessons for the track to find the lesson and navigation
  const lessons = await contentParser.getAllLessons(track as 'ai' | 'data-engineering')
  
  // Find the lesson by slug
  const lesson = lessons.find(l => l.slug === slug)
  
  if (!lesson) {
    notFound()
  }

  // Get navigation (previous/next lessons)
  const currentIndex = lessons.findIndex(l => l.slug === slug)
  const previousLesson = currentIndex > 0 ? lessons[currentIndex - 1] : undefined
  const nextLesson = currentIndex < lessons.length - 1 ? lessons[currentIndex + 1] : undefined

  return (
    <MainLayout>
      <LessonContent
        lesson={lesson}
        previousLesson={previousLesson}
        nextLesson={nextLesson}
      />
    </MainLayout>
  )
}

function LoadingFallback() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="animate-pulse">
          {/* Breadcrumbs skeleton */}
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-6"></div>
          
          {/* Header skeleton */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
            <div className="flex items-start space-x-3 mb-4">
              <div className="w-6 h-6 bg-gray-200 rounded"></div>
              <div className="flex-1">
                <div className="h-4 bg-gray-200 rounded w-1/4 mb-2"></div>
                <div className="h-6 bg-gray-200 rounded w-2/3 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/3"></div>
              </div>
            </div>
          </div>
          
          {/* Content skeleton */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="space-y-3">
              <div className="h-4 bg-gray-200 rounded w-full"></div>
              <div className="h-4 bg-gray-200 rounded w-5/6"></div>
              <div className="h-4 bg-gray-200 rounded w-4/5"></div>
              <div className="h-32 bg-gray-200 rounded w-full"></div>
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
              <div className="h-4 bg-gray-200 rounded w-2/3"></div>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function LessonPage({ params }: LessonPageProps) {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <LessonPageContent params={params} />
    </Suspense>
  )
}