function createHelloWorld() {
    return function(...args: any[]): string {
        return "Hello World";
    };
}