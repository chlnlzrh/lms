import { notFound } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { LessonContent } from '@/components/content/lesson-content'
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

  // Get track info
  const trackInfo = await contentParser.getTrackInfo(track)
  if (!trackInfo) {
    notFound()
  }

  // Get lesson data
  const lesson = await contentParser.getLesson(track, slug)
  if (!lesson) {
    notFound()
  }

  // Find the module this lesson belongs to
  const moduleNumber = lesson.moduleNumber
  const selectedModule = trackInfo.modules.find(m => m.moduleNumber === moduleNumber)

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Book of Knowledge', href: '/book-of-knowledge' },
    { label: trackInfo.title, href: `/book-of-knowledge/${track}` },
    ...(selectedModule ? [{ 
      label: selectedModule.title, 
      href: `/book-of-knowledge/${track}/module-${moduleNumber}` 
    }] : []),
    { label: lesson.frontmatter.title || lesson.title }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Lesson Content */}
        <LessonContent 
          lesson={lesson}
          track={track}
          basePath="/book-of-knowledge"
        />
      </div>
    </MainLayout>
  )
}

function LoadingFallback() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="h-12 bg-gray-200 rounded mb-6"></div>
          <div className="space-y-4">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="h-4 bg-gray-200 rounded"></div>
            ))}
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