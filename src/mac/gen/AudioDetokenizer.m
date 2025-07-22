//
// AudioDetokenizer.m
//
// This file was automatically generated and should not be edited.
//

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "AudioDetokenizer.h"

@implementation AudioDetokenizerInput

- (instancetype)initWithSemantic_tokens:(MLMultiArray *)semantic_tokens global_tokens:(MLMultiArray *)global_tokens {
    self = [super init];
    if (self) {
        _semantic_tokens = semantic_tokens;
        _global_tokens = global_tokens;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"semantic_tokens", @"global_tokens"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"semantic_tokens"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.semantic_tokens];
    }
    if ([featureName isEqualToString:@"global_tokens"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.global_tokens];
    }
    return nil;
}

@end

@implementation AudioDetokenizerOutput

- (instancetype)initWithWav_recon:(MLMultiArray *)wav_recon {
    self = [super init];
    if (self) {
        _wav_recon = wav_recon;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"wav_recon"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"wav_recon"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.wav_recon];
    }
    return nil;
}

@end

@implementation AudioDetokenizer


/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"AudioDetokenizer" ofType:@"mlmodelc"];
    if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load AudioDetokenizer.mlmodelc in the bundle resource"); return nil; }
    return [NSURL fileURLWithPath:assetPath];
}


/**
    Initialize AudioDetokenizer instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of AudioDetokenizer.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model {
    if (model == nil) {
        return nil;
    }
    self = [super init];
    if (self != nil) {
        _model = model;
    }
    return self;
}


/**
    Initialize AudioDetokenizer instance with the model in this bundle.
*/
- (nullable instancetype)init {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}


/**
    Initialize AudioDetokenizer instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
}


/**
    Initialize AudioDetokenizer instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for AudioDetokenizer.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Initialize AudioDetokenizer instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for AudioDetokenizer.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Construct AudioDetokenizer instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid AudioDetokenizer instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(AudioDetokenizer * _Nullable model, NSError * _Nullable error))handler {
    [self loadContentsOfURL:(NSURL * _Nonnull)[self URLOfModelInThisBundle]
              configuration:configuration
          completionHandler:handler];
}


/**
    Construct AudioDetokenizer instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid AudioDetokenizer instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(AudioDetokenizer * _Nullable model, NSError * _Nullable error))handler {
    [MLModel loadContentsOfURL:modelURL
                 configuration:configuration
             completionHandler:^(MLModel *model, NSError *error) {
        if (model != nil) {
            AudioDetokenizer *typedModel = [[AudioDetokenizer alloc] initWithMLModel:model];
            handler(typedModel, nil);
        } else {
            handler(nil, error);
        }
    }];
}

- (nullable AudioDetokenizerOutput *)predictionFromFeatures:(AudioDetokenizerInput *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self predictionFromFeatures:input options:[[MLPredictionOptions alloc] init] error:error];
}

- (nullable AudioDetokenizerOutput *)predictionFromFeatures:(AudioDetokenizerInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [self.model predictionFromFeatures:input options:options error:error];
    if (!outFeatures) { return nil; }
    return [[AudioDetokenizerOutput alloc] initWithWav_recon:(MLMultiArray *)[outFeatures featureValueForName:@"wav_recon"].multiArrayValue];
}

- (void)predictionFromFeatures:(AudioDetokenizerInput *)input completionHandler:(void (^)(AudioDetokenizerOutput * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            AudioDetokenizerOutput *output = [[AudioDetokenizerOutput alloc] initWithWav_recon:(MLMultiArray *)[prediction featureValueForName:@"wav_recon"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (void)predictionFromFeatures:(AudioDetokenizerInput *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(AudioDetokenizerOutput * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input options:options completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            AudioDetokenizerOutput *output = [[AudioDetokenizerOutput alloc] initWithWav_recon:(MLMultiArray *)[prediction featureValueForName:@"wav_recon"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (nullable AudioDetokenizerOutput *)predictionFromSemantic_tokens:(MLMultiArray *)semantic_tokens global_tokens:(MLMultiArray *)global_tokens error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    AudioDetokenizerInput *input_ = [[AudioDetokenizerInput alloc] initWithSemantic_tokens:semantic_tokens global_tokens:global_tokens];
    return [self predictionFromFeatures:input_ error:error];
}

- (nullable NSArray<AudioDetokenizerOutput *> *)predictionsFromInputs:(NSArray<AudioDetokenizerInput*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLBatchProvider> inBatch = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
    id<MLBatchProvider> outBatch = [self.model predictionsFromBatch:inBatch options:options error:error];
    if (!outBatch) { return nil; }
    NSMutableArray<AudioDetokenizerOutput*> *results = [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
    for (NSInteger i = 0; i < outBatch.count; i++) {
        id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
        AudioDetokenizerOutput * result = [[AudioDetokenizerOutput alloc] initWithWav_recon:(MLMultiArray *)[resultProvider featureValueForName:@"wav_recon"].multiArrayValue];
        [results addObject:result];
    }
    return results;
}

@end
