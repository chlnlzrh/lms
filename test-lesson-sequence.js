// Test script to verify lesson sequence parsing
const { contentParser } = require('./src/lib/content-parser.ts')

async function testSequencing() {
  console.log('Testing AI lesson sequencing...')
  
  try {
    const aiLessons = await contentParser.getAllLessons('ai')
    console.log(`Found ${aiLessons.length} AI lessons`)
    
    console.log('\nFirst 10 AI lessons:')
    aiLessons.slice(0, 10).forEach((lesson, index) => {
      console.log(`${index + 1}. M${lesson.moduleNumber?.toString().padStart(2, '0')}-L${lesson.lessonNumber?.toString().padStart(3, '0')} - ${lesson.frontmatter.title}`)
    })
    
    console.log('\n' + '='.repeat(50))
    console.log('Testing DE lesson sequencing...')
    
    const deLessons = await contentParser.getAllLessons('data-engineering')
    console.log(`Found ${deLessons.length} DE lessons`)
    
    console.log('\nFirst 10 DE lessons:')
    deLessons.slice(0, 10).forEach((lesson, index) => {
      console.log(`${index + 1}. M${lesson.moduleNumber?.toString().padStart(2, '0')}-L${lesson.lessonNumber?.toString().padStart(3, '0')} - ${lesson.frontmatter.title}`)
    })
    
    // Check module distribution
    console.log('\n' + '='.repeat(50))
    console.log('AI Module Distribution:')
    const aiModules = {}
    aiLessons.forEach(lesson => {
      const mod = lesson.moduleNumber || 'Unknown'
      aiModules[mod] = (aiModules[mod] || 0) + 1
    })
    Object.entries(aiModules).sort().forEach(([mod, count]) => {
      console.log(`Module ${mod}: ${count} lessons`)
    })
    
    console.log('\nDE Module Distribution:')
    const deModules = {}
    deLessons.forEach(lesson => {
      const mod = lesson.moduleNumber || 'Unknown'
      deModules[mod] = (deModules[mod] || 0) + 1
    })
    Object.entries(deModules).sort().forEach(([mod, count]) => {
      console.log(`Module ${mod}: ${count} lessons`)
    })
    
  } catch (error) {
    console.error('Error testing sequencing:', error)
  }
}

testSequencing()