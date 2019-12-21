## uml: sequence diagram
```
@startuml
    skinparam backgroundColor #EEEBDC
    skinparam handwritten false
    actor User
    boundary foo1
    control foo2
    entity foo3
    database foo4
    collections picks

    User -> "reqTag()" : enter hashtag
    "login()" -> Customer : session token
    activate "login()"
    Customer -> "placeOrder()" : session token, order info
    actor Agent
@enduml
```