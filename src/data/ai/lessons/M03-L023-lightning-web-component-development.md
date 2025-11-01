# Lightning Web Components: Modern Web Standards for Salesforce Development

## Core Concepts

Lightning Web Components (LWC) represents a fundamental shift in Salesforce development: building UI components with native web standards rather than proprietary frameworks. Unlike Aura Components (Salesforce's previous framework), LWC leverages ECMAScript 7+, Web Components standards, and modern JavaScript patterns that engineers already know from React, Vue, or vanilla JS development.

### Technical Definition

LWC is a programming model for building Salesforce UI components using standard HTML, modern JavaScript (ES6+), and CSS. The framework provides a thin layer over native Web Components APIs—Custom Elements, Shadow DOM, Templates, and ES Modules—with Salesforce-specific extensions for data access, event handling, and Lightning platform integration.

### Engineering Analogy: Framework Weight vs. Standards

**Traditional Approach (Aura Components):**

```javascript
// Aura Component Controller
({
    handleClick: function(component, event, helper) {
        var value = component.get("v.recordId");
        var action = component.get("c.getRecord");
        action.setParams({ recordId: value });
        action.setCallback(this, function(response) {
            if (response.getState() === "SUCCESS") {
                component.set("v.record", response.getReturnValue());
            }
        });
        $A.enqueueAction(action);
    }
})
```

**Modern Approach (Lightning Web Components):**

```javascript
// LWC Component JavaScript
import { LightningElement, api, wire } from 'lwc';
import getRecord from '@salesforce/apex/RecordController.getRecord';

export default class RecordDisplay extends LightningElement {
    @api recordId;
    record;
    error;

    async handleClick() {
        try {
            this.record = await getRecord({ recordId: this.recordId });
        } catch (error) {
            this.error = error;
        }
    }
}
```

The LWC version uses standard async/await, ES6 classes, decorators, and import statements. No framework-specific syntax for basic operations—just modern JavaScript. The result: 30-40% smaller bundle sizes, faster load times (measured at 2-3x faster initial render), and code that transfers directly to other frameworks.

### Why This Matters Now

Three engineering realities make LWC critical:

1. **Performance ceiling**: Aura's abstraction layer creates overhead. LWC components render 50-70% faster in benchmarks with complex DOM trees (1000+ nodes).

2. **Skills portability**: LWC knowledge transfers bidirectionally with React, Vue, and vanilla JS. No framework lock-in for your team's skillset.

3. **Maintenance burden**: Salesforce won't remove Aura, but all new platform features (like wire adapters for GraphQL) ship for LWC first or exclusively. Technical debt accumulates faster in Aura.

### Key Insight That Changes Engineering Thinking

**Stop thinking "Salesforce-specific framework." Start thinking "standard web components with Salesforce adapters."**

90% of your LWC code is portable JavaScript. The Salesforce-specific parts (@wire decorators, Lightning Data Service, platform events) are isolated imports. This mental model changes architecture decisions: build business logic in pure JavaScript modules, keep LWC components thin presentation layers, and suddenly you can unit test 80%+ of your code without Salesforce-specific test infrastructure.

## Technical Components

### 1. Component Structure and Lifecycle

LWC components consist of three files sharing a base name:

```
recordCard/
├── recordCard.js       // JavaScript class
├── recordCard.html     // HTML template
└── recordCard.css      // Component-scoped styles (optional)
```

**JavaScript Module (recordCard.js):**

```javascript
import { LightningElement, api, track } from 'lwc';

export default class RecordCard extends LightningElement {
    // Public property (exposed to parent components)
    @api recordId;
    @api objectApiName;
    
    // Private reactive property (triggers re-render when changed)
    @track recordData = {};
    
    // Computed property (recalculates when dependencies change)
    get displayTitle() {
        return this.recordData.Name || 'Unnamed Record';
    }
    
    // Lifecycle hooks
    connectedCallback() {
        // Called when component inserted into DOM
        console.log('Component mounted, recordId:', this.recordId);
    }
    
    renderedCallback() {
        // Called after every render
        // Avoid state changes here (causes render loops)
    }
    
    disconnectedCallback() {
        // Cleanup: remove listeners, clear timers
    }
    
    errorCallback(error, stack) {
        // Error boundary for child component errors
        console.error('Child error:', error, stack);
    }
}
```

**HTML Template (recordCard.html):**

```html
<template>
    <div class="card">
        <h2>{displayTitle}</h2>
        <template if:true={recordData}>
            <p>ID: {recordId}</p>
            <p>Type: {objectApiName}</p>
        </template>
        <template if:false={recordData}>
            <p>Loading...</p>
        </template>
    </div>
</template>
```

**Practical Implications:**

- **Shadow DOM encapsulation**: CSS doesn't leak out, parent styles don't leak in (except inheritable properties like font). This prevents style conflicts but complicates global theming.

- **Lifecycle sequencing**: `connectedCallback` runs before first render. DOM queries work in `renderedCallback`, not `connectedCallback`. Missing this causes null reference errors.

- **Reactive vs. non-reactive**: Only `@track`, `@api`, and `@wire` properties trigger re-renders. Plain class properties don't. Mutating array/object contents doesn't trigger re-renders unless you reassign the reference.

**Real Constraints:**

- No direct DOM manipulation in `connectedCallback` (DOM doesn't exist yet)
- `renderedCallback` runs on every render—expensive operations here kill performance
- Shadow DOM blocks global CSS selectors; must use CSS custom properties for theming
- Template directives (`if:true`, `for:each`) only work on `<template>` tags

### 2. Decorators and Reactivity System

LWC provides three core decorators that control component behavior:

```javascript
import { LightningElement, api, track, wire } from 'lwc';

export default class DataComponent extends LightningElement {
    // @api: Public property, settable by parent components
    @api maxResults = 10;
    
    // @track: Private reactive property (legacy, mostly unnecessary in modern LWC)
    @track filterState = { category: 'all', sortOrder: 'asc' };
    
    // @wire: Declarative data binding to Apex or wire adapters
    @wire(getRecords, { max: '$maxResults' })
    wiredRecords({ error, data }) {
        if (data) {
            this.records = data;
        } else if (error) {
            this.error = error;
        }
    }
    
    records = [];
    error;
    
    // Reactivity gotcha: This won't trigger re-render
    handleBadUpdate() {
        this.filterState.category = 'new';  // Mutating tracked object
    }
    
    // Correct: Reassign the reference
    handleGoodUpdate() {
        this.filterState = { ...this.filterState, category: 'new' };
    }
}
```

**Technical Explanation:**

- **@api decorator**: Creates a getter/setter pair. When parent sets the property, LWC triggers reactivity. All @api properties appear in component's public API contract.

- **@track decorator**: Largely obsolete. Modern LWC (API version 40+) automatically tracks primitive properties. Only needed for deep reactivity in objects/arrays, but immutable patterns (spread operator) are more reliable.

- **@wire decorator**: Connects to data providers. The function re-executes when reactive parameters (prefix `$`) change. Wire adapters cache results and handle loading states automatically.

**Practical Implications:**

```javascript
// Wire with dynamic parameters
@wire(getRecords, { 
    objectType: '$objectApiName',  // Reactive: re-fetch when changes
    limit: 10                      // Static: never triggers re-fetch
})
wiredData;

// Accessing wire data in template
get hasRecords() {
    return this.wiredData.data && this.wiredData.data.length > 0;
}
```

**Common Failure Mode:**

```javascript
// WRONG: Mutating tracked array doesn't trigger render
this.items.push(newItem);

// CORRECT: Create new array reference
this.items = [...this.items, newItem];

// WRONG: Nested object mutation not detected
this.config.settings.debug = true;

// CORRECT: Immutable update pattern
this.config = {
    ...this.config,
    settings: { ...this.config.settings, debug: true }
};
```

### 3. Component Communication Patterns

**Parent-to-Child (Property Binding):**

```html
<!-- parent.html -->
<template>
    <c-child-component 
        record-id={selectedId}
        max-items={itemLimit}
        onselect={handleChildSelect}>
    </c-child-component>
</template>
```

```javascript
// childComponent.js
export default class ChildComponent extends LightningElement {
    @api recordId;   // Set by parent
    @api maxItems;
    
    handleInternalClick() {
        // Child-to-Parent: Fire custom event
        this.dispatchEvent(new CustomEvent('select', {
            detail: { id: this.recordId },
            bubbles: true,    // Event propagates up DOM tree
            composed: true    // Event crosses shadow boundary
        }));
    }
}
```

**Child-to-Parent (Custom Events):**

```javascript
// parent.js
export default class Parent extends LightningElement {
    handleChildSelect(event) {
        const selectedId = event.detail.id;
        console.log('Child selected:', selectedId);
        // event.stopPropagation() to prevent further bubbling
    }
}
```

**Sibling Communication (Pub/Sub Pattern):**

```javascript
// lightningMessageService for cross-component messaging
import { LightningElement, wire } from 'lwc';
import { publish, subscribe, MessageContext } from 'lightning/messageService';
import RECORD_SELECTED_CHANNEL from '@salesforce/messageChannel/RecordSelected__c';

export default class Publisher extends LightningElement {
    @wire(MessageContext)
    messageContext;
    
    publishSelection(recordId) {
        const payload = { recordId, timestamp: Date.now() };
        publish(this.messageContext, RECORD_SELECTED_CHANNEL, payload);
    }
}

export default class Subscriber extends LightningElement {
    @wire(MessageContext)
    messageContext;
    
    subscription;
    
    connectedCallback() {
        this.subscription = subscribe(
            this.messageContext,
            RECORD_SELECTED_CHANNEL,
            (message) => this.handleMessage(message)
        );
    }
    
    handleMessage(message) {
        console.log('Received:', message.recordId);
    }
}
```

**Real Constraints:**

- Events don't cross shadow boundaries unless `composed: true`
- Lightning Message Service requires message channel metadata (XML definition)
- Custom events only communicate up the DOM tree; siblings need pub/sub or common ancestor
- Event data must be serializable (no functions, DOM nodes, or circular references)

### 4. Server Integration and Data Management

**Imperative Apex Calls:**

```javascript
// Apex controller with @AuraEnabled annotation
import getAccountData from '@salesforce/apex/AccountController.getAccountData';

export default class AccountView extends LightningElement {
    @api accountId;
    accountData;
    error;
    isLoading = false;
    
    async loadAccount() {
        this.isLoading = true;
        try {
            // Imperative call returns a Promise
            this.accountData = await getAccountData({ 
                accountId: this.accountId 
            });
            this.error = undefined;
        } catch (error) {
            this.error = this.reduceErrors(error);
            this.accountData = undefined;
        } finally {
            this.isLoading = false;
        }
    }
    
    reduceErrors(error) {
        // Apex errors come in various formats
        if (Array.isArray(error.body)) {
            return error.body.map(e => e.message).join(', ');
        } else if (error.body?.message) {
            return error.body.message;
        }
        return 'Unknown error';
    }
}
```

**Wire Adapter Pattern:**

```javascript
import { LightningElement, api, wire } from 'lwc';
import { getRecord, getFieldValue } from 'lightning/uiRecordApi';
import NAME_FIELD from '@salesforce/schema/Account.Name';
import REVENUE_FIELD from '@salesforce/schema/Account.AnnualRevenue';

export default class RecordViewer extends LightningElement {
    @api recordId;
    
    // Wire adapter automatically refetches when recordId changes
    @wire(getRecord, { 
        recordId: '$recordId', 
        fields: [NAME_FIELD, REVENUE_FIELD] 
    })
    record;
    
    get name() {
        return getFieldValue(this.record.data, NAME_FIELD);
    }
    
    get revenue() {
        return getFieldValue(this.record.data, REVENUE_FIELD);
    }
}
```

**Lightning Data Service (LDS) vs. Apex:**

| Aspect | Lightning Data Service | Apex Controller |
|--------|----------------------|-----------------|
| Caching | Automatic client-side cache | Manual implementation |
| Security | Automatic CRUD/FLS enforcement | Manual with `WITH SECURITY_ENFORCED` |
| Network | Optimized batch requests | Individual calls |
| Flexibility | Limited to standard CRUD | Full query control |
| Performance | ~100ms (cached) | ~300-500ms (server round-trip) |

**Practical Decision Framework:**

Use LDS (`lightning/uiRecordApi`) when:
- Reading/writing single records
- Standard field access patterns
- Leveraging shared cache across components

Use Apex when:
- Complex queries (joins, aggregations)
- Business logic on server
- Bulk operations
- Non-standard security requirements

### 5. Styling and CSS Encapsulation

**Component CSS (recordCard.css):**

```css
/* Scoped to this component automatically */
.card {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 16px;
    background: var(--lwc-colorBackground);  /* Use CSS custom properties */
}

.card h2 {
    color: var(--lwc-colorTextDefault);
    margin: 0 0 12px 0;
}

/* Styling slotted content from parent */
::slotted(p) {
    font-style: italic;
}

/* Host element styling */
:host {
    display: block;
    margin: 8px;
}
```

**Breaking Shadow DOM Encapsulation (when necessary):**

```javascript
// In renderedCallback, after initial render
renderedCallback() {
    if (this.hasRendered) return;
    this.hasRendered = true;
    
    // Access shadow DOM internals
    const style = document.createElement('style');
    style.innerText = `
        .external-library-class {
            color: red !important;
        }
    `;
    this.template.appendChild(style);
}
```

**SLDS (Salesforce Lightning Design System) Integration:**

```html
<template>
    <!-- Use SLDS utility classes -->
    <div class="slds-card slds-p-around_medium">
        <button class="slds-button slds-button_brand" onclick={handleClick}>
            Click Me
        </button>
    </div>
</template>
```

**Real Constraints:**

- Shadow DOM blocks parent selectors (`.parent .child` won't work across boundaries)
- Only inheritable CSS properties (font, color) penetrate shadow boundary
- SLDS classes must be explicitly included; not automatic
- Third-party CSS libraries often fail due to shadow DOM isolation

## Hands-On Exercises

### Exercise 1: Build a Reactive Counter with Parent-Child Communication

**